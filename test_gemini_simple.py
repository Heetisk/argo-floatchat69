import requests
import os
import json

# API Key from the curl command (or env)
# The user provided key: AIzaSyCougLN0dv552OyB3_E0sLeaucfOJ1JuPA
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCougLN0dv552OyB3_E0sLeaucfOJ1JuPA")

def test_gemini():
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Explain how AI works in a few words"
                    }
                ]
            }
        ]
    }
    
    print(f"Testing Gemini API with model: gemini-2.0-flash")
    print(f"URL: {url}")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("\nStatus Code:", response.status_code)
        print("Response:\n", json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print("Response content:", e.response.text)

if __name__ == "__main__":
    test_gemini()
