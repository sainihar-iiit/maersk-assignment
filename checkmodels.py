import google.generativeai as genai

genai.configure(api_key="AIzaSyArWsznqkiFQzinwHvyzY6viihtGW1MOlY")

print("🔍 Available Gemini models for your key:\n")
for m in genai.list_models():
    print(m.name)
