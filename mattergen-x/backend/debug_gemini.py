
import os
import google.generativeai as genai

KEY = "AIzaSyC0r4JXIWORxTAUHNxN0MkbvaGlI-DFsjQ" # First key from .env.local

print(f"Checking key: ...{KEY[-5:]}")

try:
    genai.configure(api_key=KEY)
    
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
    # Try a test generation
    print("\nAttempting test generation with 'gemini-pro'...")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello")
    print(f"Success! Response: {response.text}")

except Exception as e:
    print(f"\nERROR: {e}")
