import streamlit as st
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from typing import Dict, Any, Optional, List
from difflib import get_close_matches

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load environment variables
load_dotenv()

class OrderProcessor:
    def __init__(self, products_file: str = 'products.json'):
        """
        Initialize the order processing system.
        
        Args:
            products_file (str): Path to products JSON file
        """
        # Load products
        try:
            with open(products_file, 'r') as f:
                self.PRODUCTS = json.load(f)
        except FileNotFoundError:
            logging.error(f"Products file '{products_file}' not found.")
            self.PRODUCTS = []

        # Configure AI model
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logging.error(f"AI model initialization error: {e}")
            self.model = None

        # Initialize session state
        if 'cart' not in st.session_state:
            st.session_state.cart = {}
        if 'order_confirmed' not in st.session_state:
            st.session_state.order_confirmed = False  # track if order has been confirmed

    def generate_intent_prompt(self, user_input: str) -> str:
        """
        Generate AI prompt for multi-action parsing.
        
        We define synonyms for each action so the model can catch them more easily.
        We also instruct the model how to handle out-of-stock or large quantity scenarios
        and how to handle potentially conflicting instructions.
        """
        products_list = ", ".join([p['name'] for p in self.PRODUCTS])

        return f"""
        You are an assistant that processes store orders for the user.
        The user may specify multiple or conflicting actions in a single request.
        You should parse them all in order.

        **Possible Actions & Synonyms**:
        1) ADD
           - synonyms: "add", "put", "place", "include", "add to cart"
        2) REMOVE
           - synonyms: "remove", "delete", "take out", "subtract"
        3) VIEW_CART
           - synonyms: "show cart", "view cart", "what's in my cart"
        4) VIEW_PRODUCTS
           - synonyms: "list products", "show products", "catalog"
        5) CONFIRM_ORDER
           - synonyms: "confirm", "checkout", "bill me", "finalize", "complete purchase", "pay"
        6) CANCEL_ORDER
           - synonyms: "cancel order", "void order", "abort order"
        7) RESET_CART
           - synonyms: "reset cart", "clear cart", "empty cart"

        **Handling Quantities**:
        - If user requests more items than in stock, you can still parse the intent normally,
          but note the quantity might exceed availability in the "explanation".

        **JSON OUTPUT**:
        Return a JSON object with a top-level key "actions" which is an array of action objects.
        Each action object has this structure:
        {{
          "intent": "ADD" | "REMOVE" | "VIEW_CART" | "VIEW_PRODUCTS" | "CONFIRM_ORDER" | "CANCEL_ORDER" | "RESET_CART",
          "items": [
            {{
              "product": "string or null",
              "quantity": number or null
            }}
            ...
          ],
          "explanation": "brief interpretation or note about conflicts/out-of-stock"
        }}

        Example:
        {{
          "actions": [
            {{
              "intent": "ADD",
              "items": [{{"product": "hemp seeds", "quantity": 2}}],
              "explanation": "Adding hemp seeds"
            }},
            {{
              "intent": "REMOVE",
              "items": [{{"product": "almond", "quantity": 1}}],
              "explanation": "Removing 1 almond"
            }},
            ...
          ]
        }}

        **Important**:
        1) If multiple actions appear, return multiple objects in "actions".
        2) If the action doesn't require items (like viewing the cart, confirming, cancelling), set "items": [].
        3) Do not wrap the JSON in code fences (like ```json).
        4) The user might mention multiple items for ADD/REMOVE.
        5) If user instructions conflict, interpret them in the order given or note it in the explanation.
        6) If user request doesn't match known actions, you can set "actions": [] or "intent": "UNKNOWN".

        **Available Products**: {products_list}

        **User Input**: "{user_input}"

        Return only valid JSON.
        """

    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """Parse user intent using AI (multi-action)."""
        if not self.model:
            # Fallback if model isn't loaded
            logging.error("AI model is not configured. Returning fallback result.")
            return {"actions": [
                {"intent": "ERROR", "items": [], "explanation": "Model not configured."}
            ]}

        try:
            response = self.model.generate_content(self.generate_intent_prompt(user_input))
            raw_text = response.candidates[0].content.parts[0].text.strip()
            
            # In case the model returns triple-backtick code blocks
            if raw_text.startswith("```") and raw_text.endswith("```"):
                raw_text = raw_text.strip("```").strip()
            
            data = json.loads(raw_text)

            # Ensure we have an "actions" key
            if "actions" not in data or not isinstance(data["actions"], list):
                data["actions"] = []

            return data
        
        except Exception as e:
            logging.error(f"Intent parsing error: {e}")
            return {
                "actions": [
                    {
                        "intent": "ERROR",
                        "items": [],
                        "explanation": f"Parsing error: {e}"
                    }
                ]
            }
    
    def find_product(self, product_query: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Advanced fuzzy matching to find the closest matching product.

        1) Try exact substring match (case-insensitive).
        2) If that fails, use difflib.get_close_matches to find the best approximate match.
        """
        if not product_query:
            return None
        
        # 1) Check for direct substring matches
        direct_matches = [
            p for p in self.PRODUCTS
            if product_query.lower() in p['name'].lower()
        ]
        if len(direct_matches) == 1:
            return direct_matches[0]
        elif len(direct_matches) > 1:
            st.warning(f"Multiple products matched '{product_query}'. Be more specific:")
            for product in direct_matches:
                st.write(f"- {product['name']}")
            return None

        # 2) Fuzzy matching (difflib)
        product_names = [p['name'] for p in self.PRODUCTS]
        close_candidates = get_close_matches(product_query, product_names, n=1, cutoff=0.6)
        if close_candidates:
            best_match_name = close_candidates[0]
            # Retrieve the product with this matching name
            for p in self.PRODUCTS:
                if p['name'] == best_match_name:
                    return p

        return None  # No good match found

    def add_to_cart(self, product_name: str, quantity: int = 1) -> bool:
        """Add a single product to cart (with stock-check)."""
        product = self.find_product(product_name)
        
        if not product:
            st.error(f"Product '{product_name}' not found.")
            return False
        
        if product['stock'] < quantity:
            st.warning(f"Requested {quantity} x '{product['name']}', but only {product['stock']} available.")
            # You might decide to add the max available or do nothing. Let's do partial add or just do nothing:
            # For demonstration, let's do partial add of whatever is in stock:
            quantity = product['stock']

        if quantity <= 0:
            st.warning(f"No '{product['name']}' added to cart.")
            return False

        # Update cart and stock
        if product['name'] in st.session_state.cart:
            st.session_state.cart[product['name']]['quantity'] += quantity
        else:
            st.session_state.cart[product['name']] = {
                'price': product['price'],
                'quantity': quantity
            }
        
        product['stock'] -= quantity
        st.success(f"Added {quantity} x '{product['name']}' to cart.")
        return True
    
    def remove_from_cart(self, product_name: str, quantity: int = 1) -> bool:
        """Remove a single product from cart (stock is restored)."""
        product = self.find_product(product_name)
        
        if not product or product['name'] not in st.session_state.cart:
            st.error(f"Product '{product_name}' not in cart.")
            return False
        
        cart_item = st.session_state.cart[product['name']]
        
        if quantity >= cart_item['quantity']:
            # Remove entire item
            removed_qty = cart_item['quantity']
            del st.session_state.cart[product['name']]
            product['stock'] += removed_qty
            st.success(f"Removed all '{product['name']}' from cart.")
        else:
            cart_item['quantity'] -= quantity
            product['stock'] += quantity
            st.success(f"Removed {quantity} x '{product['name']}' from cart.")
        
        return True
    
    def view_cart(self) -> None:
        """Display cart contents."""
        st.subheader("Your Cart")
        if not st.session_state.cart:
            st.info("Cart is empty.")
            return
        
        total = 0
        for name, details in st.session_state.cart.items():
            subtotal = details['price'] * details['quantity']
            total += subtotal
            st.write(f"{name}: ${details['price']:.2f} x {details['quantity']} = ${subtotal:.2f}")
        
        st.write(f"**Total: ${total:.2f}**")
    
    def view_products(self) -> None:
        """List available products."""
        st.subheader("Available Products")
        for product in self.PRODUCTS:
            st.write(f"{product['name']} - ${product['price']:.2f} (Stock: {product['stock']})")
    
    def confirm_order(self) -> None:
        """Process final order."""
        if not st.session_state.cart:
            st.info("No items to confirm.")
            return
        st.session_state.order_confirmed = True
        st.success("Order confirmed! Thank you for your purchase. (Cart is now locked in.)")
        # We keep the cart items as is, or we could clear them. 
        # For demonstration, let's just keep them so user can see what was confirmed.

    def cancel_order(self) -> None:
        """
        Cancel the confirmed order.
        In real scenarios, you might have to handle partial refunds or revert states.
        Here, we'll just unlock the cart and restore items to stock as if the order never happened.
        """
        if not st.session_state.order_confirmed:
            st.info("No confirmed order to cancel.")
            return

        # Move items from cart back to product stock
        for name, details in st.session_state.cart.items():
            # find product in self.PRODUCTS
            for p in self.PRODUCTS:
                if p['name'] == name:
                    p['stock'] += details['quantity']
                    break

        st.session_state.cart = {}
        st.session_state.order_confirmed = False
        st.warning("Confirmed order has been canceled and cart cleared.")

    def reset_cart(self) -> None:
        """
        Clear the cart entirely (regardless of confirmed or not).
        Restore stock to all items in the cart.
        """
        if not st.session_state.cart:
            st.info("Cart is already empty.")
            return
        
        # Restore items to stock
        for name, details in st.session_state.cart.items():
            for p in self.PRODUCTS:
                if p['name'] == name:
                    p['stock'] += details['quantity']
                    break

        st.session_state.cart = {}
        st.info("Cart has been reset (emptied).")

    def process_request(self, user_input: str) -> None:
        """Main request processing method: handle multiple actions in order."""
        parsed_data = self.parse_intent(user_input)
        actions = parsed_data.get("actions", [])

        # Helper to handle a single action
        def handle_action(intent: str, items: List[Dict[str, Any]], explanation: str):
            logging.info(f"Action: {intent} | Items: {items} | Explanation: {explanation}")
            
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

            elif intent == "ERROR":
                st.error(explanation or "Unknown error.")

            else:
                st.warning(f"Unsupported or unknown intent: {intent}")

        # Process each action in turn
        for action in actions:
            intent = action.get("intent", "UNKNOWN")
            items = action.get("items", [])
            explanation = action.get("explanation", "")
            handle_action(intent, items, explanation)

def main():
    st.title("GreenLife Order Processing")

    # Initialize processor
    processor = OrderProcessor()
    
    # User input
    user_input = st.text_input("Enter your request:", "")
    process_button = st.button("Process")
    
    if process_button and user_input.strip():
        with st.spinner("Processing your request..."):
            processor.process_request(user_input)
    
    # Always show current cart at the bottom
    st.write("---")
    processor.view_cart()

if __name__ == "__main__":
    main()
