import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
from google import genai

client = genai.Client(api_key=api_key)
content="How do I improve my webpage https://yourembeddedsoftwaredesigner.com?" 
result = client.models.generate_content(model="gemini-2.5-flash",contents=content)
print(result.text)
print("Prompt Tokens:",result.usage_metadata.prompt_token_count)
print("Response Tokens:",result.usage_metadata.candidates_token_count)
