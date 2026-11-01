
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found!")
else:
    client = genai.Client(api_key=api_key)

    text = """
    He go to school yesterday.
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=f"Correct the grammar: {text}",
        config={
            "http_options": {
                "timeout": 180000
            }
        }
    )

    print("Gemini API connected successfully!")
    print(response.text)
