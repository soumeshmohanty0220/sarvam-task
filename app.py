import streamlit as st
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from typing import Dict, Any, Optional

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
            logging.error(f"Products file {products_file} not found")
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
    
    def generate_intent_prompt(self, user_input: str) -> str:
        """Generate AI prompt for intent parsing."""
        return f"""
        Precisely parse user's order request:
        Available Products: {', '.join([p['name'] for p in self.PRODUCTS])}
        User Input: "{user_input}"

        Return JSON with:
        {{
            "intent": "ADD/REMOVE/VIEW_CART/VIEW_PRODUCTS/CONFIRM_ORDER",
            "product": "exact product name or null",
            "quantity": number or null,
            "explanation": "brief interpretation"
        }}
        """
    
    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """Parse user intent using AI."""
        try:
            response = self.model.generate_content(
                self.generate_intent_prompt(user_input)
            )
            raw_text = response.candidates[0].content.parts[0].text.strip()
            
            # Clean JSON if wrapped in code block
            if raw_text.startswith("```json") and raw_text.endswith("```"):
                raw_text = raw_text[7:-3].strip()
            
            return json.loads(raw_text)
        
        except Exception as e:
            logging.error(f"Intent parsing error: {e}")
            return {"intent": "ERROR", "explanation": str(e)}
    
    def find_product(self, product_query: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fuzzy product matching."""
        if not product_query:
            return None
        
        matches = [
            p for p in self.PRODUCTS 
            if product_query.lower() in p['name'].lower()
        ]
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            st.warning("Multiple products found. Be more specific:")
            for product in matches:
                st.write(f"- {product['name']}")
        
        return None
    
    def add_to_cart(self, product_name: str, quantity: int = 1) -> bool:
        """Add product to cart."""
        product = self.find_product(product_name)
        
        if not product:
            st.error(f"Product '{product_name}' not found.")
            return False
        
        if product['stock'] < quantity:
            st.error(f"Insufficient stock. Available: {product['stock']}")
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
        st.success(f"Added {quantity} {product['name']} to cart.")
        return True
    
    def remove_from_cart(self, product_name: str, quantity: int = 1) -> bool:
        """Remove product from cart."""
        product = self.find_product(product_name)
        
        if not product or product['name'] not in st.session_state.cart:
            st.error(f"Product '{product_name}' not in cart.")
            return False
        
        cart_item = st.session_state.cart[product['name']]
        
        if quantity >= cart_item['quantity']:
            del st.session_state.cart[product['name']]
            product['stock'] += cart_item['quantity']
            st.success(f"Removed all {product['name']} from cart.")
        else:
            cart_item['quantity'] -= quantity
            product['stock'] += quantity
            st.success(f"Removed {quantity} {product['name']} from cart.")
        
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
            st.write(f"{name}: ${details['price']} x {details['quantity']} = ${subtotal:.2f}")
        
        st.write(f"**Total: ${total:.2f}**")
    
    def view_products(self) -> None:
        """List available products."""
        st.subheader("Available Products")
        for product in self.PRODUCTS:
            st.write(f"{product['name']} - ${product['price']} (Stock: {product['stock']})")
    
    def confirm_order(self) -> None:
        """Process final order."""
        if not st.session_state.cart:
            st.info("No items to confirm.")
            return
        
        st.success("Order confirmed! Thank you for your purchase.")
        st.session_state.cart = {}
    
    def process_request(self, user_input: str) -> None:
        """Main request processing method."""
        intent_data = self.parse_intent(user_input)
        intent = intent_data.get('intent', 'UNKNOWN')
        
        intent_handlers = {
            'ADD': lambda: self.add_to_cart(
                intent_data.get('product'), 
                intent_data.get('quantity', 1)
            ),
            'REMOVE': lambda: self.remove_from_cart(
                intent_data.get('product'), 
                intent_data.get('quantity', 1)
            ),
            'VIEW_CART': self.view_cart,
            'VIEW_PRODUCTS': self.view_products,
            'CONFIRM_ORDER': self.confirm_order,
            'ERROR': lambda: st.error(intent_data.get('explanation', 'Unknown error'))
        }
        
        handler = intent_handlers.get(intent)
        if handler:
            handler()
        else:
            st.warning(f"Unsupported intent: {intent}")

def main():
    st.title("GreenLife Order Processing")
    
    # Initialize processor
    processor = OrderProcessor()
    
    # User input
    user_input = st.text_input("Enter request:", "")
    process_button = st.button("Process")
    
    if process_button and user_input:
        with st.spinner("Processing request..."):
            processor.process_request(user_input)
    
    # Always show current cart
    st.write("---")
    processor.view_cart()

if __name__ == "__main__":
    main()