import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536


def get_embedding(text: str) -> list[float]:
    text = text.strip().replace("\n", " ")
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    texts = [t.strip().replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    test_text = "What is the average revenue per region?"
    embedding = get_embedding(test_text)
    print(f"Text: {test_text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Model: {EMBEDDING_MODEL}")