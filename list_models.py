
import os
import requests
import json

API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCougLN0dv552OyB3_E0sLeaucfOJ1JuPA")
URL = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

def list_models():
    print(f"Listing models...")
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Found {len(models)} models.")
            for m in models:
                if 'generateContent' in m.get('supportedGenerationMethods', []):
                     print(f"- {m['name']} ({m.get('displayName')})")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_models()
