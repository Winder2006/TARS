import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def test_openai_connection():
    try:
        # Test the API with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, this is a test."}
            ]
        )
        print("OpenAI API connection successful!")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error connecting to OpenAI API:", str(e))

if __name__ == "__main__":
    test_openai_connection() 