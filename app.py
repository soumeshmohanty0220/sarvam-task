import streamlit as st
import json
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from typing import Dict, Any, Optional, List
from difflib import get_close_matches
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --------------------------------------------------------------------------
# 1) CONFIG & UTILS
# --------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
load_dotenv()

ORDERS_DB_FILE = "orders_db.json"
PRODUCTS_DB_FILE = "products_db.json"

def load_orders_db() -> List[Dict[str, Any]]:
    if not os.path.exists(ORDERS_DB_FILE):
        return []
    with open(ORDERS_DB_FILE, "r") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                return []
            return data
        except json.JSONDecodeError:
            return []

def save_orders_db(all_orders: List[Dict[str, Any]]):
    with open(ORDERS_DB_FILE, "w") as f:
        json.dump(all_orders, f, indent=2)

def load_products_db() -> List[Dict[str, Any]]:
    if not os.path.exists(PRODUCTS_DB_FILE):
        return []
    with open(PRODUCTS_DB_FILE, "r") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                return []
            return data
        except json.JSONDecodeError:
            return []

def save_products_db(all_products: List[Dict[str, Any]]):
    with open(PRODUCTS_DB_FILE, "w") as f:
        json.dump(all_products, f, indent=2)

# --------------------------------------------------------------------------
# 2) VECTOR STORE SETUP
# --------------------------------------------------------------------------

class VectorStore:
    EMBEDDINGS_FILE = "product_embeddings.npy"
    FAISS_INDEX_FILE = "faiss_index.bin"

    def __init__(self, products: List[Dict[str, Any]], embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.products = products
        self.model = SentenceTransformer(embedding_model_name)
        
        if self._check_existing_files():
            self.embeddings = self._load_embeddings()
            self.index = self._load_faiss_index()
        else:
            self.embeddings = self._generate_embeddings()
            self.index = self._create_faiss_index()
            self._save_embeddings()
            self._save_faiss_index()

    def _check_existing_files(self) -> bool:
        return os.path.exists(self.EMBEDDINGS_FILE) and os.path.exists(self.FAISS_INDEX_FILE)

    def _generate_embeddings(self) -> np.ndarray:
        product_texts = [product['name'] for product in self.products]
        embeddings = self.model.encode(product_texts, convert_to_numpy=True)
        return embeddings

    def _create_faiss_index(self) -> faiss.Index:
        if len(self.embeddings) == 0:
            dimension = 384  # Default dimension for MiniLM embeddings
            index = faiss.IndexFlatL2(dimension)
            return index
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index

    def _save_embeddings(self):
        np.save(self.EMBEDDINGS_FILE, self.embeddings)
        logging.info("Embeddings saved to disk.")

    def _load_embeddings(self) -> np.ndarray:
        embeddings = np.load(self.EMBEDDINGS_FILE)
        logging.info("Embeddings loaded from disk.")
        return embeddings

    def _save_faiss_index(self):
        faiss.write_index(self.index, self.FAISS_INDEX_FILE)
        logging.info("FAISS index saved to disk.")

    def _load_faiss_index(self) -> faiss.Index:
        index = faiss.read_index(self.FAISS_INDEX_FILE)
        logging.info("FAISS index loaded from disk.")
        return index

    def query(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.products:
            return []
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        print(query_embedding)
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.products):
                results.append(self.products[idx])
        print(results)
        return results

# --------------------------------------------------------------------------
# 3) MAIN ORDER PROCESSOR CLASS
# --------------------------------------------------------------------------

class OrderProcessor:
    def __init__(self, products_file: str = 'products.json'):
        self.PRODUCTS = load_products_db()
        if not self.PRODUCTS:
            try:
                with open(products_file, 'r') as f:
                    self.PRODUCTS = json.load(f)
                save_products_db(self.PRODUCTS)
            except FileNotFoundError:
                logging.error(f"Products file '{products_file}' not found. Proceeding with empty list.")
                self.PRODUCTS = []
        self.vector_store = VectorStore(self.PRODUCTS)  # Uses the updated VectorStore
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash-8b")
        except Exception as e:
            logging.error(f"AI model initialization error: {e}")
            self.model = None
        self.all_orders = load_orders_db()

    def generate_intent_prompt(self, user_input: str) -> str:
        relevant_products = self.vector_store.query(user_input, top_k=20)
        products_list = ", ".join(f"{p['name']} - ${p['price']} (Stock: {p['stock']})" for p in relevant_products)
        prompt = (f"You are an assistant that processes store orders for the user (with a user_id).\n"
              "The user may specify multiple or combined actions in a single request.\n"
              "You should parse them all in order. Each action should become a separate\n"
              "object in the 'actions' array.\n\n"
              "**Possible Actions & Synonyms**:\n"
              "1) ADD\n"
              "   - synonyms: 'add', 'put', 'place', 'include', 'add to cart'\n"
              "2) REMOVE\n"
              "   - synonyms: 'remove', 'delete', 'take out', 'subtract'\n"
              "3) VIEW_CART\n"
              "   - synonyms: 'show cart', 'view cart', 'what's in my cart', 'see my cart', 'show my current orders'\n"
              "4) VIEW_PRODUCTS\n"
              "   - synonyms: 'list products', 'show products', 'catalog'\n"
              "5) CONFIRM_ORDER\n"
              "   - synonyms: 'confirm', 'checkout', 'bill me', 'finalize', 'complete purchase',\n"
              "     'pay', 'complete my order', 'place order', 'place an order'\n"
              "6) CANCEL_ORDER\n"
              "   - synonyms: 'cancel order', 'void order', 'abort order'\n"
              "7) RESET_CART\n"
              "   - synonyms: 'reset cart', 'clear cart', 'empty cart'\n"
              "8) REPEAT_ORDER\n"
              "   - synonyms: 'repeat my last order', 'repeat my last 3 orders', 'repeat all my orders',\n"
              "     'reorder', 'order again', etc.\n"
              "   Explanation:\n"
              "     If the user references 'repeat all my previous orders', or 'repeat last X orders',\n"
              "     you can interpret:\n"
              "       'intent': 'REPEAT_ORDER',\n"
              "       'items': [{'product': null, 'quantity': null}],\n"
              "       'explanation': 'User wants to reorder from all or last X.'\n\n"
              "**Handling Multiple Items**:\n"
              "- If the user says 'add 2 quinoa and 2 almonds', you can group them under a single\n"
              "  ADD action with multiple items in the array, or use multiple ADD actions. Either is valid.\n"
              "  E.g.:\n"
              "  {\n"
              "    'actions': [\n"
              "      {\n"
              "        'intent': 'ADD',\n"
              "        'items': [\n"
              "          {'product': 'quinoa', 'quantity': 2},\n"
              "          {'product': 'almonds', 'quantity': 2}\n"
              "        ],\n"
              "        'explanation': 'User wants multiple items'\n"
              "      }\n"
              "    ]\n"
              "  }\n"
              "- If the user says: 'Place an order for 2 quinoa and 2 almonds',\n"
              "  interpret as:\n"
              "  {\n"
              "    'actions': [\n"
              "      {\n"
              "        'intent': 'ADD',\n"
              "        'items': [\n"
              "          {'product': 'quinoa', 'quantity': 2},\n"
              "          {'product': 'almonds', 'quantity': 2}\n"
              "        ],\n"
              "        'explanation': 'User wants to add these items'\n"
              "      },\n"
              "      {\n"
              "        'intent': 'CONFIRM_ORDER',\n"
              "        'items': [],\n"
              "        'explanation': 'User wants to finalize/checkout'\n"
              "      }\n"
              "    ]\n"
              "  }\n\n"
              "**JSON OUTPUT**:\n"
              "Return JSON of the form:\n"
              "{\n"
              "  'actions': [\n"
              "    {\n"
              "      'intent': 'ADD or REMOVE or ...',\n"
              "      'items': [\n"
              "        {\n"
              "          'product': 'string or null',\n"
              "          'quantity': number or null\n"
              "        },\n"
              "        ...\n"
              "      ],\n"
              "      'explanation': 'string'\n"
              "    },\n"
              "    ...\n"
              "  ]\n"
              "}\n\n"
              "If no recognized store action is found, return:\n"
              "{\n"
              "  'actions': [\n"
              "    {\n"
              "      'intent': 'UNKNOWN',\n"
              "      'items': [],\n"
              "      'explanation': 'No recognized store intent found.'\n"
              "    }\n"
              "  ]\n"
              "}\n\n"
              "**Available Products**: {products_list}\n\n"
              f"**User Input**: \"{user_input}\"\n\n"
              "Return only valid JSON, without backticks or code fences.")
    
        return prompt

    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        if not self.model:
            logging.error("AI model is not configured. Returning fallback result.")
            return {
                "actions": [
                    {
                        "intent": "ERROR",
                        "items": [],
                        "explanation": "AI model not configured."
                    }
                ]
            }

        try:
            prompt = self.generate_intent_prompt(user_input)
            response = self.model.generate_content(prompt)

            if not response.candidates or not response.candidates[0].content.parts:
                logging.error("Empty or missing response from the AI.")
                return {
                    "actions": [
                        {
                            "intent": "UNKNOWN",
                            "items": [],
                            "explanation": "No recognized store intent or empty response."
                        }
                    ]
                }

            raw_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            logging.debug(f"Raw AI output: {raw_text}")

            data = json.loads(raw_text)
            if "actions" not in data or not isinstance(data["actions"], list):
                data["actions"] = []

            return data

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return {
                "actions": [
                    {
                        "intent": "ERROR",
                        "items": [],
                        "explanation": f"Parsing error: {e}"
                    }
                ]
            }
        except Exception as e:
            logging.error(f"Parsing error: {e}")
            return {
                "actions": [
                    {
                        "intent": "ERROR",
                        "items": [],
                        "explanation": f"Parsing error: {e}"
                    }
                ]
            }

    def generate_general_response(self, user_input: str) -> str:
        if not self.model:
            return "I'm sorry, I'm having trouble connecting to my conversation module right now."

        relevant_products = self.vector_store.query(user_input, top_k=10)
        product_descriptions = "\n".join([
            f"- {p['name']}: ${p['price']} (Stock: {p['stock']})"
            for p in relevant_products
        ]) if relevant_products else "No relevant products found."

        prompt = f"""
        You are a friendly assistant for GreenLife, a health food store.
        The user is asking a question or making a statement that might not map to a store action.
        Please respond conversationally. If they ask about products, refer to them from this list:

        {product_descriptions}

        If it's a general question, answer politely or provide small talk.

        User said: "{user_input}"
        """

        try:
            response = self.model.generate_content(prompt)
            if not response.candidates or not response.candidates[0].content.parts:
                return "I'm sorry, I didn't get any response from the conversation model."

            full_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
            return full_text

        except Exception as e:
            logging.error(f"General response error: {e}")
            return "I'm sorry, I had trouble generating a conversational response right now."

    def find_product(self, product_query: Optional[str]) -> Optional[Dict[str, Any]]:
        if not product_query:
            return None

        direct_matches = [p for p in self.PRODUCTS if product_query.lower() in p['name'].lower()]
        if len(direct_matches) == 1:
            return direct_matches[0]
        elif len(direct_matches) > 1:
            st.warning(f"Multiple products matched '{product_query}'. Please be more specific:")
            for product in direct_matches:
                st.markdown(f"- {product['name']}")
            return None

        product_names = [p['name'] for p in self.PRODUCTS]
        close_candidates = get_close_matches(product_query, product_names, n=1, cutoff=0.6)
        if close_candidates:
            best_match_name = close_candidates[0]
            for p in self.PRODUCTS:
                if p['name'] == best_match_name:
                    return p

        return None

    def add_to_cart(self, product_name: str, quantity: int = 1) -> bool:
        product = self.find_product(product_name)
        if not product:
            st.error(f"❌ Sorry, I can't find a product called '{product_name}'.")
            return False

        if product['stock'] < quantity:
            st.warning(
                f"You requested {quantity} of '{product['name']}', "
                f"but only {product['stock']} are in stock. Adjusting accordingly."
            )
            quantity = product['stock']

        if quantity <= 0:
            st.warning(f"⚠️ '{product['name']}' is out of stock, so none added.")
            return False

        if product['name'] in st.session_state.cart:
            st.session_state.cart[product['name']]['quantity'] += quantity
        else:
            st.session_state.cart[product['name']] = {
                'price': product['price'],
                'quantity': quantity
            }

        product['stock'] -= quantity
        save_products_db(self.PRODUCTS)  # Save updated stock
        st.success(f"✅ Added {quantity} x '{product['name']}' to your cart.")
        return True

    def remove_from_cart(self, product_name: str, quantity: int = 1) -> bool:
        product = self.find_product(product_name)
        if not product or product['name'] not in st.session_state.cart:
            st.error(f"❌ '{product_name}' isn't in your cart.")
            return False

        cart_item = st.session_state.cart[product['name']]
        if quantity >= cart_item['quantity']:
            removed_qty = cart_item['quantity']
            del st.session_state.cart[product['name']]
            product['stock'] += removed_qty
            st.success(f"✅ Removed all of '{product['name']}' from your cart.")
        else:
            cart_item['quantity'] -= quantity
            product['stock'] += quantity
            st.success(f"✅ Removed {quantity} of '{product['name']}'.")

        save_products_db(self.PRODUCTS)  # Save updated stock
        return True

    def view_cart(self) -> None:
        st.markdown("### 🛒 **Your Cart:**")
        if not st.session_state.cart:
            st.info("🛍️ Your cart is empty right now.")
            return

        total = 0
        for name, details in st.session_state.cart.items():
            subtotal = details['price'] * details['quantity']
            total += subtotal
            st.markdown(f"- **{name}**: ${details['price']:.2f} × {details['quantity']} = ${subtotal:.2f}")
        st.markdown(f"**🧾 Total: ${total:.2f}**")

    def view_products(self) -> None:
        st.markdown("### 📦 **Available Products:**")
        if not self.PRODUCTS:
            st.info("📭 No products available at the moment.")
            return
        for product in self.PRODUCTS:
            st.markdown(f"• **{product['name']}** — ${product['price']:.2f} (Stock: {product['stock']})")

    def confirm_order(self) -> None:
        if not st.session_state.cart:
            st.info("🛒 Your cart is empty, so there's nothing to confirm.")
            return
        if not st.session_state.user_id:
            st.warning("⚠️ Please enter a user ID or name first.")
            return

        completed_items = {
            name: {
                "price": details["price"],
                "quantity": details["quantity"]
            }
            for name, details in st.session_state.cart.items()
        }

        order_id = str(uuid.uuid4())
        new_order = {
            "order_id": order_id,
            "user_id": st.session_state.user_id,
            "timestamp": datetime.now().isoformat(),
            "items": completed_items
        }

        self.all_orders.append(new_order)
        save_orders_db(self.all_orders)

        st.session_state.order_confirmed = True
        st.success(f"✅ **Order #{order_id}** confirmed for user '{st.session_state.user_id}'!")
        st.session_state.cart = {}
        save_products_db(self.PRODUCTS)  # Save updated stock

    def cancel_order(self) -> None:
        if not st.session_state.order_confirmed:
            st.info("🛠️ No confirmed order to cancel in this session.")
            return

        for name, details in st.session_state.cart.items():
            product = self.find_product(name)
            if product:
                product['stock'] += details['quantity']

        st.session_state.cart = {}
        st.session_state.order_confirmed = False
        st.warning("⚠️ Your previously confirmed order (this session) is canceled and the cart is cleared.")
        save_products_db(self.PRODUCTS)  # Save updated stock

    def reset_cart(self) -> None:
        if not st.session_state.cart:
            st.info("🧹 Your cart is already empty.")
            return

        for name, details in st.session_state.cart.items():
            product = self.find_product(name)
            if product:
                product['stock'] += details['quantity']

        st.session_state.cart = {}
        st.info("🧹 Your cart has been reset.")
        save_products_db(self.PRODUCTS)  # Save updated stock

    def repeat_order_by_criteria(self, explanation: str) -> None:
        if not st.session_state.user_id:
            st.warning("⚠️ We don't know who you are. Please enter a user ID or name above.")
            return

        user_orders = [o for o in self.all_orders if o["user_id"] == st.session_state.user_id]
        if not user_orders:
            st.info("📭 You have no previous orders in our records.")
            return

        lower_explanation = explanation.lower()

        if "all" in lower_explanation:
            for order in user_orders:
                self._add_order_to_cart(order)
            st.success("✅ All your previous orders have been added to the cart!")
            return

        match = re.search(r"last\s+(\d+)\s+orders?", lower_explanation)
        if match:
            x = int(match.group(1))
            user_orders_sorted = sorted(user_orders, key=lambda o: o["timestamp"])
            subset = user_orders_sorted[-x:]
            for order in subset:
                self._add_order_to_cart(order)
            st.success(f"✅ Re-added your last {x} orders.")
            return

        user_orders_sorted = sorted(user_orders, key=lambda o: o["timestamp"])
        last_order = user_orders_sorted[-1]
        self._add_order_to_cart(last_order)
        st.success("✅ Re-added your most recent order.")

    def _add_order_to_cart(self, order: Dict[str, Any]):
        for product_name, details in order["items"].items():
            qty = details["quantity"]
            self.add_to_cart(product_name, qty)

    def process_request(self, user_input: str) -> None:
        parsed_data = self.parse_intent(user_input)
        actions = parsed_data.get("actions", [])
        recognized_intent = False

        def handle_action(intent: str, items: List[Dict[str, Any]], explanation: str):
            nonlocal recognized_intent
            if intent in [
                "ADD", "REMOVE", "VIEW_CART", "VIEW_PRODUCTS",
                "CONFIRM_ORDER", "CANCEL_ORDER", "RESET_CART", "REPEAT_ORDER"
            ]:
                recognized_intent = True

            logging.info(f"Handling Intent: {intent} | Items: {items} | Explanation: {explanation}")

            if intent == "ADD":
                for item in items:
                    product_name = item.get('product')
                    quantity = item.get('quantity', 1)
                    if product_name:
                        self.add_to_cart(product_name, quantity)

            elif intent == "REMOVE":
                for item in items:
                    product_name = item.get('product')
                    quantity = item.get('quantity', 1)
                    if product_name:
                        self.remove_from_cart(product_name, quantity)

            elif intent == "VIEW_CART":
                self.view_cart()

            elif intent == "VIEW_PRODUCTS":
                self.view_products()

            elif intent == "CONFIRM_ORDER":
                self.confirm_order()

            elif intent == "CANCEL_ORDER":
                self.cancel_order()

            elif intent == "RESET_CART":
                self.reset_cart()

            elif intent == "REPEAT_ORDER":
                self.repeat_order_by_criteria(explanation)

            elif intent == "ERROR":
                st.error(f"❌ **ERROR:** {explanation or 'Unknown parsing error.'}")

            elif intent == "UNKNOWN":
                pass
            else:
                st.warning(f"⚠️ Unrecognized action: '{intent}'")

        for action in actions:
            intent = action.get("intent", "UNKNOWN")
            items = action.get("items", [])
            explanation = action.get("explanation", "")
            handle_action(intent, items, explanation)

        if not recognized_intent:
            response_text = self.generate_general_response(user_input)
            st.markdown(f"**Bot:** {response_text}")

# --------------------------------------------------------------------------
# 4) STREAMLIT ENTRY POINT
# --------------------------------------------------------------------------

# Remove the @st.cache_resource decorator to prevent caching issues
def get_order_processor() -> OrderProcessor:
    return OrderProcessor()

def main():
    st.set_page_config(page_title="GreenLife Chatbot", page_icon="🌿", layout="wide")
    
    # The dark theme is handled via the config.toml file, so no need for additional CSS
    # Remove or comment out any existing inline CSS that sets background colors
    
    st.sidebar.title("🌿 GreenLife Chatbot")
    st.sidebar.markdown("**Your Health Food Companion**")
    
    # Initialize session state variables
    if 'cart' not in st.session_state:
        st.session_state.cart = {}
    if 'order_confirmed' not in st.session_state:
        st.session_state.order_confirmed = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    # User ID Input in Sidebar
    with st.sidebar.form(key='user_form'):
        user_id = st.text_input("Enter your User ID or Name:", "")
        submit_user = st.form_submit_button(label='Set User ID')
        if submit_user and user_id.strip():
            st.session_state.user_id = user_id.strip()
            st.sidebar.success(f"User ID set to: **{st.session_state.user_id}**")

    processor = get_order_processor()

    # Main Interface
    st.title("🌿 GreenLife Chatbot")
    st.markdown("Welcome to **GreenLife**, your friendly health food store assistant! How can I help you today?")

    # User Input Form
    with st.form(key='chat_form'):
        user_input = st.text_input("💬 Ask me anything or place an order:", "")
        submit = st.form_submit_button(label='Send')

    if submit and user_input.strip():
        with st.spinner("🤖 Processing your request..."):
            processor.process_request(user_input.strip())
        st.success("✅ Your request has been processed!")

    st.markdown("---")

    # Display Cart and Products using Columns and Expanders
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.expander("📦 Available Products", expanded=True):
            processor.view_products()

    with col2:
        with st.expander("🛒 Your Cart", expanded=True):
            processor.view_cart()
            
            
if __name__ == "__main__":
    main()
