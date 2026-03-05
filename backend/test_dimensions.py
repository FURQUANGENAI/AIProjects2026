import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
model = "models/gemini-embedding-001"

embeddings = GoogleGenerativeAIEmbeddings(
    model=model,
    google_api_key=google_api_key
)

text = "This is a test query."
vector = embeddings.embed_query(text)

print(f"Model: {model}")
print(f"Vector dimension: {len(vector)}")
