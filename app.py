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
import re  # For parsing "last X orders"

# --------------------------------------------------------------------------
# 1) CONFIG & UTILS
# --------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
load_dotenv()  # Load environment variables (for GEMINI_API_KEY, etc.)

ORDERS_DB_FILE = "orders_db.json"

def load_orders_db() -> List[Dict[str, Any]]:
    """
    Loads all orders from a JSON file (orders_db.json).
    If the file does not exist or is invalid, returns an empty list.
    """
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
    """
    Saves the entire list of orders (all_orders) to the JSON file (orders_db.json).
    Overwrites any existing content.
    """
    with open(ORDERS_DB_FILE, "w") as f:
        json.dump(all_orders, f, indent=2)

# --------------------------------------------------------------------------
# 2) MAIN ORDER PROCESSOR CLASS
# --------------------------------------------------------------------------

class OrderProcessor:
    """
    The OrderProcessor class wraps all functionality for:
    - Managing a product catalog (loaded from products.json)
    - Handling store actions (ADD, REMOVE, VIEW_CART, etc.)
    - Confirming orders and saving them to a JSON database (orders_db.json)
    - Repeating past orders (all or last X)
    - Falling back to a general conversation if no store action is found
    """

    def __init__(self, products_file: str = 'products.json'):
        """
        Initialize the order processing system.
        Loads products from a JSON file and configures the AI model.
        Also initializes or loads the orders database.
        """
        # 2.1) Load products from JSON
        try:
            with open(products_file, 'r') as f:
                self.PRODUCTS = json.load(f)
        except FileNotFoundError:
            logging.error(f"Products file '{products_file}' not found. Proceeding with empty list.")
            self.PRODUCTS = []

        # 2.2) Configure the AI model (Google Generative AI, Gemini)
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash-8b")
        except Exception as e:
            logging.error(f"AI model initialization error: {e}")
            self.model = None

        # 2.3) Initialize Streamlit session states (cart, user info, etc.)
        if 'cart' not in st.session_state:
            st.session_state.cart = {}
        if 'order_confirmed' not in st.session_state:
            st.session_state.order_confirmed = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None

        # 2.4) Load the existing orders database from JSON
        self.all_orders = load_orders_db()

    # ----------------------------------------------------------------------
    # 3) LLM PROMPTS AND PARSING
    # ----------------------------------------------------------------------

    def generate_intent_prompt(self, user_input: str) -> str:
        """
        Constructs a prompt that instructs the AI to parse store-related actions
        from the user's text. Includes synonyms and the new REPEAT_ORDER feature
        for "repeat all my orders" or "last X orders".
        We also provide examples to ensure the model understands multiple actions
        can appear in a single request.

        If the user says something like:
          - "Add 2 quinoa and show me the cart"
            => parse as two actions: ADD + VIEW_CART
          - "Place an order for 2 quinoa and 2 almonds"
            => parse as two actions: ADD + CONFIRM_ORDER
          - "Add 1 kale, 2 almonds, 3 quinoa, remove 1 quinoa, and confirm order"
            => parse as ADD + REMOVE + CONFIRM_ORDER
        """

        products_list = ", ".join(p['name'] for p in self.PRODUCTS)

        return f"""
        You are an assistant that processes store orders for the user (with a user_id).
        The user may specify multiple or combined actions in a single request.
        You should parse them all in order. Each action should become a separate
        object in the "actions" array.

        **Possible Actions & Synonyms**:
        1) ADD
           - synonyms: "add", "put", "place", "include", "add to cart"
        2) REMOVE
           - synonyms: "remove", "delete", "take out", "subtract"
        3) VIEW_CART
           - synonyms: "show cart", "view cart", "what's in my cart", "see my cart", "show my current orders"
        4) VIEW_PRODUCTS
           - synonyms: "list products", "show products", "catalog"
        5) CONFIRM_ORDER
           - synonyms: "confirm", "checkout", "bill me", "finalize", "complete purchase",
             "pay", "complete my order", "place order", "place an order"
        6) CANCEL_ORDER
           - synonyms: "cancel order", "void order", "abort order"
        7) RESET_CART
           - synonyms: "reset cart", "clear cart", "empty cart"
        8) REPEAT_ORDER
           - synonyms: "repeat my last order", "repeat my last 3 orders", "repeat all my orders",
             "reorder", "order again", etc.
           Explanation:
             If the user references "repeat all my previous orders", or "repeat last X orders",
             you can interpret:
               "intent": "REPEAT_ORDER",
               "items": [{{"product": null, "quantity": null}}],
               "explanation": "User wants to reorder from all or last X."

        **Handling Multiple Items**:
        - If the user says "add 2 quinoa and 2 almonds", you can group them under a single
          ADD action with multiple items in the array, or use multiple ADD actions. Either is valid.
          E.g.:
          {{
            "actions": [
              {{
                "intent": "ADD",
                "items": [
                  {{"product": "quinoa", "quantity": 2}},
                  {{"product": "almonds", "quantity": 2}}
                ],
                "explanation": "User wants multiple items"
              }}
            ]
          }}

        **Handling Quantities**:
        - If user requests more items than in stock, parse it normally,
          but note the potential issue in "explanation" if needed.

        **Examples**:
        - If the user says: "Add 2 quinoa and show me the cart",
          return:
          {{
            "actions": [
              {{
                "intent": "ADD",
                "items": [
                  {{"product": "quinoa", "quantity": 2}}
                ],
                "explanation": "User wants to add quinoa"
              }},
              {{
                "intent": "VIEW_CART",
                "items": [],
                "explanation": "User wants to see the cart"
              }}
            ]
          }}
        - If the user says: "Place an order for 2 quinoa and 2 almonds",
          interpret as:
          {{
            "actions": [
              {{
                "intent": "ADD",
                "items": [
                  {{"product": "quinoa", "quantity": 2}},
                  {{"product": "almonds", "quantity": 2}}
                ],
                "explanation": "User wants to add these items"
              }},
              {{
                "intent": "CONFIRM_ORDER",
                "items": [],
                "explanation": "User wants to finalize/checkout"
              }}
            ]
          }}

        **JSON OUTPUT**:
        Return JSON of the form:
        {{
          "actions": [
            {{
              "intent": "ADD or REMOVE or ...",
              "items": [
                {{
                  "product": "string or null",
                  "quantity": number or null
                }},
                ...
              ],
              "explanation": "string"
            }},
            ...
          ]
        }}

        If no recognized store action is found, return:
        {{
          "actions": [
            {{
              "intent": "UNKNOWN",
              "items": [],
              "explanation": "No recognized store intent found."
            }}
          ]
        }}

        **Available Products**: {products_list}

        **User Input**: "{user_input}"

        Return only valid JSON, without backticks or code fences.
        """

    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Calls the AI model to parse the user's text into a JSON structure
        containing one or more actions. If we can't parse JSON or the model
        isn't available, we return a fallback dict with an ERROR or UNKNOWN.
        """
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
            # Generate structured content from the prompt
            response = self.model.generate_content(self.generate_intent_prompt(user_input))

            # Check if we got a valid response with content
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

            # Combine text from all parts
            raw_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
            # Clean up any code fences
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            logging.debug(f"Raw AI output: {raw_text}")

            # Attempt to parse JSON from the model's response
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

    # ----------------------------------------------------------------------
    # 4) GENERAL CONVERSATION FALLBACK
    # ----------------------------------------------------------------------

    def generate_general_response(self, user_input: str) -> str:
        """
        If the user input doesn't match store actions, we respond conversationally
        using a simpler open-ended prompt. We still include product listings
        for context, in case they ask about them.
        """
        if not self.model:
            return "I'm sorry, I'm having trouble connecting to my conversation module right now."

        product_descriptions = "\n".join([
            f"- {p['name']}: ${p['price']} (Stock: {p['stock']})"
            for p in self.PRODUCTS
        ])

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

    # ----------------------------------------------------------------------
    # 5) BASIC STORE ACTIONS
    # ----------------------------------------------------------------------

    def find_product(self, product_query: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Fuzzy or substring match to find the best matching product from self.PRODUCTS.
        If multiple matches, warns the user to be more specific.
        If none, returns None.
        """
        if not product_query:
            return None

        # Substring match
        direct_matches = [p for p in self.PRODUCTS if product_query.lower() in p['name'].lower()]
        if len(direct_matches) == 1:
            return direct_matches[0]
        elif len(direct_matches) > 1:
            st.warning(f"Multiple products matched '{product_query}'. Please be more specific:")
            for product in direct_matches:
                st.write(f"- {product['name']}")
            return None

        # Fuzzy match using difflib
        product_names = [p['name'] for p in self.PRODUCTS]
        close_candidates = get_close_matches(product_query, product_names, n=1, cutoff=0.6)
        if close_candidates:
            best_match_name = close_candidates[0]
            for p in self.PRODUCTS:
                if p['name'] == best_match_name:
                    return p

        return None

    def add_to_cart(self, product_name: str, quantity: int = 1) -> bool:
        """
        Adds a product to the cart if it's in stock.
        If the requested quantity exceeds stock, we adjust it.
        If out of stock, we do nothing.
        """
        product = self.find_product(product_name)
        if not product:
            st.error(f"Sorry, I can't find a product called '{product_name}'.")
            return False

        if product['stock'] < quantity:
            st.warning(
                f"You requested {quantity} of '{product['name']}', "
                f"but only {product['stock']} are in stock. Adjusting accordingly."
            )
            quantity = product['stock']

        if quantity <= 0:
            st.warning(f"'{product['name']}' is out of stock, so none added.")
            return False

        # Update cart
        if product['name'] in st.session_state.cart:
            st.session_state.cart[product['name']]['quantity'] += quantity
        else:
            st.session_state.cart[product['name']] = {
                'price': product['price'],
                'quantity': quantity
            }

        # Decrease product stock
        product['stock'] -= quantity

        st.success(f"Added {quantity} x '{product['name']}' to your cart.")
        return True

    def remove_from_cart(self, product_name: str, quantity: int = 1) -> bool:
        """
        Removes a product from the cart. If the quantity to remove is equal or
        greater than what's in the cart, it removes that product completely.
        """
        product = self.find_product(product_name)
        if not product or product['name'] not in st.session_state.cart:
            st.error(f"'{product_name}' isn't in your cart.")
            return False

        cart_item = st.session_state.cart[product['name']]
        if quantity >= cart_item['quantity']:
            removed_qty = cart_item['quantity']
            del st.session_state.cart[product['name']]
            product['stock'] += removed_qty
            st.success(f"Removed all of '{product['name']}' from your cart.")
        else:
            cart_item['quantity'] -= quantity
            product['stock'] += quantity
            st.success(f"Removed {quantity} of '{product['name']}'.")
        return True

    def view_cart(self) -> None:
        """
        Displays the current cart contents and total cost.
        If the cart is empty, informs the user.
        """
        st.subheader("Your Cart:")
        if not st.session_state.cart:
            st.info("Your cart is empty right now.")
            return

        total = 0
        for name, details in st.session_state.cart.items():
            subtotal = details['price'] * details['quantity']
            total += subtotal
            st.write(f"- **{name}**: ${details['price']:.2f} × {details['quantity']} = ${subtotal:.2f}")
        st.write(f"**Total: ${total:.2f}**")

    def view_products(self) -> None:
        """
        Shows the list of available products (name, price, stock).
        If no products are loaded, it notifies the user.
        """
        st.subheader("Available Products:")
        if not self.PRODUCTS:
            st.info("No products available at the moment.")
            return
        for product in self.PRODUCTS:
            st.write(f"• **{product['name']}** — ${product['price']:.2f} (Stock: {product['stock']})")

    def confirm_order(self) -> None:
        """
        Confirms the current cart:
         - Generates a unique order_id
         - Saves to the orders database with user_id, timestamp, and items
         - Clears the cart
        """
        if not st.session_state.cart:
            st.info("Your cart is empty, so there's nothing to confirm.")
            return
        if not st.session_state.user_id:
            st.warning("Please enter a user ID or name first.")
            return

        # Take a snapshot of the cart
        completed_items = {
            name: {
                "price": details["price"],
                "quantity": details["quantity"]
            }
            for name, details in st.session_state.cart.items()
        }

        # Create a new order record
        order_id = str(uuid.uuid4())
        new_order = {
            "order_id": order_id,
            "user_id": st.session_state.user_id,
            "timestamp": datetime.now().isoformat(),
            "items": completed_items
        }

        # Save in memory
        self.all_orders.append(new_order)
        # Persist to disk
        save_orders_db(self.all_orders)

        st.session_state.order_confirmed = True
        st.success(f"Order #{order_id} confirmed for user '{st.session_state.user_id}'!")
        st.session_state.cart = {}  # Clear the cart

    def cancel_order(self) -> None:
        """
        Cancels a previously confirmed order in this session (if any),
        restoring items to stock. Clears the cart as well.
        """
        if not st.session_state.order_confirmed:
            st.info("No confirmed order to cancel in this session.")
            return

        for name, details in st.session_state.cart.items():
            product = self.find_product(name)
            if product:
                product['stock'] += details['quantity']

        st.session_state.cart = {}
        st.session_state.order_confirmed = False
        st.warning("Your previously confirmed order (this session) is canceled and the cart is cleared.")

    def reset_cart(self) -> None:
        """
        Resets the cart at any time (regardless of whether the order is confirmed),
        restoring items to stock.
        """
        if not st.session_state.cart:
            st.info("Your cart is already empty.")
            return

        for name, details in st.session_state.cart.items():
            product = self.find_product(name)
            if product:
                product['stock'] += details['quantity']

        st.session_state.cart = {}
        st.info("Your cart has been reset.")

    # ----------------------------------------------------------------------
    # 6) REPEAT ORDER LOGIC
    # ----------------------------------------------------------------------

    def repeat_order_by_criteria(self, explanation: str) -> None:
        """
        Interprets the LLM's "explanation" to see if the user wants to repeat
        "all" orders or the "last X orders." Then re-adds them to the cart.
        If none of those patterns are found, re-add the single most recent order.
        """
        if not st.session_state.user_id:
            st.warning("We don't know who you are. Please enter a user ID or name above.")
            return

        # Filter orders for the current user
        user_orders = [o for o in self.all_orders if o["user_id"] == st.session_state.user_id]
        if not user_orders:
            st.info("You have no previous orders in our records.")
            return

        lower_explanation = explanation.lower()

        # Case A: "repeat all my orders"
        if "all" in lower_explanation:
            for order in user_orders:
                self._add_order_to_cart(order)
            st.success("All your previous orders have been added to the cart!")
            return

        # Case B: "repeat last X orders"
        match = re.search(r"last\s+(\d+)\s+orders?", lower_explanation)
        if match:
            x = int(match.group(1))
            user_orders_sorted = sorted(user_orders, key=lambda o: o["timestamp"])
            subset = user_orders_sorted[-x:]  # last X
            for order in subset:
                self._add_order_to_cart(order)
            st.success(f"Re-added your last {x} orders.")
            return

        # Case C: no match => repeat single most recent
        user_orders_sorted = sorted(user_orders, key=lambda o: o["timestamp"])
        last_order = user_orders_sorted[-1]
        self._add_order_to_cart(last_order)
        st.success("Re-added your most recent order.")

    def _add_order_to_cart(self, order: Dict[str, Any]):
        """
        Helper method to re-add items from a previous order record into the current
        cart (and reduce product stock).
        """
        for product_name, details in order["items"].items():
            qty = details["quantity"]
            self.add_to_cart(product_name, qty)

    # ----------------------------------------------------------------------
    # 7) REQUEST PROCESSING: ACTIONS + FALLBACK
    # ----------------------------------------------------------------------

    def process_request(self, user_input: str) -> None:
        """
        Main request flow:
         - Parse the user's text for store actions.
         - If recognized, handle them in order (e.g., ADD, REMOVE).
         - If no recognized actions, fallback to general conversation.
        """
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
                st.error(f"ERROR: {explanation or 'Unknown parsing error.'}")

            elif intent == "UNKNOWN":
                # We'll handle the fallback after processing all actions
                pass

            else:
                st.warning(f"Unrecognized action: '{intent}'")

        # Process each action returned by the model in sequence
        for action in actions:
            intent = action.get("intent", "UNKNOWN")
            items = action.get("items", [])
            explanation = action.get("explanation", "")
            handle_action(intent, items, explanation)

        # If no recognized store action, fallback to general conversation
        if not recognized_intent:
            response_text = self.generate_general_response(user_input)
            st.write(response_text)

# --------------------------------------------------------------------------
# 8) STREAMLIT ENTRY POINT
# --------------------------------------------------------------------------

def main():
    """
    The Streamlit entry point.
    1) Prompts the user for a user ID/name.
    2) Initializes the OrderProcessor.
    3) Accepts user input in a text box.
    4) Displays the cart contents after each request.
    """
    st.title("GreenLife Chatbot")

    # Prompt for user ID or name
    user_id = st.text_input("Enter your user ID or name:", "")
    if user_id:
        st.session_state.user_id = user_id

    # Initialize the OrderProcessor
    processor = OrderProcessor()

    # Let the user type a request
    user_input = st.text_input("Ask me anything or place an order:")
    if st.button("Send") and user_input.strip():
        with st.spinner("Thinking..."):
            processor.process_request(user_input)

    st.write("---")
    processor.view_cart()

# Run the Streamlit app
if __name__ == "__main__":
    main()
