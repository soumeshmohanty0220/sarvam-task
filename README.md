# GreenLife Chatbot — Order Processing with Repeat Feature

This repository contains a **Streamlit**-based chatbot that can:

1. **Process store orders** (adding and removing items, viewing and confirming the cart).  
2. **Store past orders** in a JSON database.  
3. **Repeat** previous orders when requested (e.g., “repeat all my orders” or “repeat my last 2 orders”).  
4. **Fall back to a general chat** if no recognized store action is found.

It leverages **Google’s Generative AI (Gemini)** 
## Features

- **ADD** items to cart (with fuzzy matching for product names).  
- **REMOVE** items from cart.  
- **VIEW_CART** or **VIEW_PRODUCTS** to check items.  
- **CONFIRM_ORDER** to finalize your purchase (stores a record in `orders_db.json`).  
- **CANCEL_ORDER** to revert a confirmed order in the current session.  
- **RESET_CART** to clear the cart at any time.  
- **REPEAT_ORDER** to re-add past orders:
  - *“Repeat all my orders”* – re-adds everything from your user history.  
  - *“Repeat my last 3 orders”* – re-adds only the last 3.  
- **General Conversation** fallback using the same LLM (if no store action is recognized).

---

## Requirements

1. **Python 3.7+**  
2. **Streamlit** (for the web UI)  
3. **google-generativeai** (Gemini / PaLM library)  
4. **python-dotenv** (for loading environment variables)  

Example:

```bash
pip install streamlit google-generativeai python-dotenv
```

You also need:

- A **Google Generative AI API key** (GEMINI_API_KEY) in your `.env` file.
- A **`products.json`** file specifying product data (example below).

---

## Setup

1. **Clone** this repository or place the files in your working directory.  
2. **Create a `.env` file** with your Gemini API key:
   ```bash
   GEMINI_API_KEY="your-google-generativeai-key-here"
   ```
3. **Create (or edit) `products.json`** to define your store inventory. For example:
   ```json
   [
     {
       "name": "Quinoa",
       "price": 5.99,
       "stock": 100
     },
     {
       "name": "Chia Seeds",
       "price": 3.49,
       "stock": 50
     },
     {
       "name": "Almonds",
       "price": 9.99,
       "stock": 200
     }
   ]
   ```
4. (Optional) **Check or empty `orders_db.json`** if you want a fresh database for storing orders.

---

## Running the Chatbot

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually install `streamlit`, `google-generativeai`, `python-dotenv`, etc.)*

2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```
   - Replace `app.py` with whatever filename contains your chatbot code.

3. **Open your browser** at the URL displayed (usually `http://localhost:8501`).

---

## How to Use

1. **Enter your User ID** at the top of the app (e.g. “john_doe”).  
2. **Type a request** (e.g., “Add 2 quinoa to cart”) and click **Send**.  
3. **View or confirm your cart** with requests like “view cart” or “confirm order.”  
4. **Repeat** your past orders with phrases like:
   - “Repeat my last order”  
   - “Repeat my last 3 orders”  
   - “Repeat all my orders”  
5. If the LLM **doesn’t recognize** a store action, it will **respond** in a general conversational style.

---

## Example Commands

- **Add items**:  
  - “Add 2 quinoa and 3 chia seeds to my cart”  
  - “Add almonds to cart”  
- **View**:  
  - “What’s in my cart?”  
  - “Show products”  
- **Confirm & Cancel**:  
  - “Confirm my order now”  
  - “Cancel my order”  
- **Repeat**:  
  - “Repeat my last order”  
  - “Repeat all previous orders”  

---

## Folder Structure

```
.
├── app.py               # Main Streamlit app (chatbot code)
├── products.json        # Store inventory
├── orders_db.json       # Saved orders database
├── .env                 # Environment variables (contains GEMINI_API_KEY)
├── requirements.txt     # Dependencies (Streamlit, google-generativeai, python-dotenv, etc.)
└── README.md            # This README
```

---

## Notes

- **Stock Updates**: When you add or remove items, the stock in `products.json` is updated in-memory. If you restart the app, `products.json` is reloaded from disk.  
- **Persisted Orders**: Each confirmed order is stored in `orders_db.json` with a unique UUID, user ID, timestamp, and item details.  
- **Fuzzy Matching**: The system tries its best to handle partial matches for product names (e.g. “Quin” → “Quinoa”).  
- **Local Environment**: Ensure your `.env` file has a valid `GEMINI_API_KEY`.

### Enjoy the GreenLife Chatbot!

Feel free to adapt, extend, or customize it for your own use cases. If you have any questions or run into issues, open an issue or contact the repository owner.
