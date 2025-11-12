# project.py
import pandas as pd
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr

# ------------------ Step 1: Load and merge all datasets ------------------
base_path = r"C:\Users\pc\Downloads\archive (1)"  # 🔧 Update if needed

# Load datasets
orders = pd.read_csv(base_path + r"\olist_orders_dataset.csv")
order_items = pd.read_csv(base_path + r"\olist_order_items_dataset.csv")
products = pd.read_csv(base_path + r"\olist_products_dataset.csv")
customers = pd.read_csv(base_path + r"\olist_customers_dataset.csv")
payments = pd.read_csv(base_path + r"\olist_order_payments_dataset.csv")
reviews = pd.read_csv(base_path + r"\olist_order_reviews_dataset.csv")
sellers = pd.read_csv(base_path + r"\olist_sellers_dataset.csv")
categories = pd.read_csv(base_path + r"\product_category_name_translation.csv")

# Merge product with category translation
products = products.merge(categories, on="product_category_name", how="left")

# Combine all datasets into one large DataFrame
merged = order_items.merge(products, on="product_id", how="left")
merged = merged.merge(sellers, on="seller_id", how="left")
merged = merged.merge(orders, on="order_id", how="left")
merged = merged.merge(customers, on="customer_id", how="left")
merged = merged.merge(payments, on="order_id", how="left")
merged = merged.merge(reviews, on="order_id", how="left")

merged.to_csv("merged.csv", index=False)
print("✅ Full merged.csv created successfully!")

# ------------------ Step 2: Configure Gemini ------------------
genai.configure(api_key="AIzaSyArWsznqkiFQzinwHvyzY6viihtGW1MOlY")  # 🔑 Replace with your actual Gemini API key
model = genai.GenerativeModel("models/gemini-2.5-flash")




# ------------------ Step 3: Initialize speech and TTS ------------------
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

# ------------------ Step 4: Define query processor ------------------
def process_query(query, merged):
    query = query.lower()

    # 🧮 Core queries from your previous version
    if "total sales" in query or "total revenue" in query:
        total = merged['price'].sum()
        return f"The total sales revenue is ₹{total:,.2f}"

    elif "total orders" in query:
        return f"The total number of orders is {merged['order_id'].nunique()}"

    elif "top" in query and "customers" in query:
        top_customers = (
            merged.groupby('customer_id')['price']
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        return f"Top 5 customers by spending:\n{top_customers.to_string()}"

    elif "top" in query and "products" in query:
        top_products = (
            merged.groupby('product_id')['price']
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        return f"Top 5 best-selling products:\n{top_products.to_string()}"

    elif "average price" in query:
        avg_price = merged['price'].mean()
        return f"The average product price is ₹{avg_price:,.2f}"

    elif "state" in query or "city" in query:
        top_states = (
            merged.groupby('customer_state')['order_id']
            .nunique()
            .sort_values(ascending=False)
            .head(5)
        )
        return f"Top 5 states with the highest orders:\n{top_states.to_string()}"

    # 🧠 Extended queries (multi-table analytics)
    elif "payment" in query:
        top_methods = merged['payment_type'].value_counts().head(5)
        return f"Top payment methods:\n{top_methods.to_string()}"

    elif "review" in query or "rating" in query:
        avg_rating = merged['review_score'].mean()
        return f"Average review score across all orders: {avg_rating:.2f}"

    elif "seller" in query:
        top_sellers = (
            merged.groupby('seller_id')['price']
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        return f"Top 5 sellers by total sales:\n{top_sellers.to_string()}"

    elif "category" in query:
        top_categories = (
            merged.groupby('product_category_name_english')['price']
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        return f"Top 5 categories by total sales:\n{top_categories.to_string()}"

    elif "delivery time" in query:
        merged['order_delivered_customer_date'] = pd.to_datetime(merged['order_delivered_customer_date'])
        merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
        merged['delivery_time_days'] = (merged['order_delivered_customer_date'] - merged['order_purchase_timestamp']).dt.days
        avg_time = merged['delivery_time_days'].mean()
        return f"Average delivery time: {avg_time:.2f} days"

    elif "stop" in query or "exit" in query:
        return "stop"

    else:

        try:
        
            prompt = f"You are a data analyst. Based on this E-commerce dataset, answer this question: {query}"
            response = model.generate_content(prompt)
            return response.text if hasattr(response, "text") else "Sorry, I couldn't generate a proper response."
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return "Sorry, Gemini AI is currently unavailable. Please try a different question or check your API key."


# ------------------ Step 5: Main chat loop ------------------
speak("Hi, I am your AI Assistant for E-commerce insights!")

while True:
    with sr.Microphone() as source:
        print("\nListening for your question... (say 'stop' to exit)")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("You said:", command)

        response_text = process_query(command, merged)

        if response_text == "stop":
            speak("Goodbye!")
            break
        else:
            speak(response_text)

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand your voice.")
        speak("Sorry, I couldn't understand.")
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        speak("Network error occurred.")

print("✅ Assistant closed successfully.")

