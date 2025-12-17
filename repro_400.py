
import os
import requests
import json

API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCougLN0dv552OyB3_E0sLeaucfOJ1JuPA")
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def send_request(payload, test_name):
    print(f"\n--- Testing: {test_name} ---")
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }
    try:
        response = requests.post(URL, headers=headers, json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code >= 400:
            print("Error Response:", response.text)
        else:
            print("Success")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Test 1: Empty Text
    payload_empty = {
        "contents": [{"parts": [{"text": ""}]}],
        "generationConfig": {"temperature": 0.1}
    }
    send_request(payload_empty, "Empty Text")

    # Test 2: Invalid Model Param (unlikely, would be 400 or 404)
    # But here we are testing the payload structure.
    
    # Test 3: Null text
    payload_null = {
        "contents": [{"parts": [{"text": None}]}],
    }
    send_request(payload_null, "Null Text")

