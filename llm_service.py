from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "env", ".env"))

sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"Sambanova API Key: {sambanova_api_key}")
print(f"OpenAI API Key: {openai_api_key}")