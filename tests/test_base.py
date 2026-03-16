from agents.base import call_claude

def test_api_connection():
    result = call_claude("Say: API connection successful. Nothing else.")
    assert "successful" in result.lower()
    print("PASS:", result)

if __name__ == "__main__":
    test_api_connection()