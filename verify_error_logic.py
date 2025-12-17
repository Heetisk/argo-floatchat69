
def simulate_error_handling(error_message):
    print(f"Testing error: {error_message}")
    try:
        # Simulate the error
        raise Exception(error_message)
    except Exception as e:
        error_str = str(e)
        # Logic from app.py
        if "429" in error_str or "Too Many Requests" in error_str or "quota" in error_str.lower():
            error_msg = (
                "⚠️ **API Limit Reached**\n\n"
                "Our api not working properly please create your own gemini api at "
                "https://aistudio.google.com/ and enter it in api key section manually"
            )
            print("MATCHED: 429 Logic triggered successfully")
            print("Message:", error_msg)
        elif "API key not valid" in error_str or "API_KEY_INVALID" in error_str:
            error_msg = (
                "❌ **Invalid API Key**\n\n"
                "The provided API Key is invalid. If you are using the System Key, it might be incorrect. "
                "Please enter a valid API Key in the sidebar."
            )
            print("MATCHED: 400 Invalid Key Logic triggered successfully")
            print("Message:", error_msg)
        else:
            print("FAILED: Generic error logic triggered")

if __name__ == "__main__":
    # Test Case 1: Standard 429 string
    simulate_error_handling("429 Client Error: Too Many Requests")
    
    # Test Case 2: Quota string
    simulate_error_handling("Quota exceeded for quota metric")
    
    # Test Case 3: Invalid API Key
    simulate_error_handling("400 Client Error: Bad Request... API key not valid")
    
    # Test Case 4: Random error
    simulate_error_handling("Random network error")
