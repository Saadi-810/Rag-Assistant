from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv  
from sqlalchemy import text
 # <-- add this
load_dotenv()                   # <-- add this

from db.database import get_vectorstore, engine
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Vector store & embeddings
vectorstore = get_vectorstore()
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)



# Mistral API configuration
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL_ID = "mistral-small"  # we just confirmed this works with curl
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set in environment")

class QueryRequest(BaseModel):
    question: str
    conversation_id: str

def fetch_conversation_memory(conversation_id: str, top_k: int = 3):
    """Fetch last few stored memories from DB"""
    with engine.connect() as conn:
        stmt = text(
            "SELECT text FROM conversation_memory "
            "WHERE conversation_id = :cid "
            "ORDER BY created_at DESC "
            "LIMIT :top_k"
        )
        result = conn.execute(stmt, {"cid": conversation_id, "top_k": top_k})
        rows = [row[0] for row in result]
    return "\n\n".join(rows)

def store_conversation_memory(conversation_id: str, answer_text: str, embedding):
    """Store new memory in DB"""
    if not isinstance(embedding, list):
        embedding = list(embedding)

    insert_stmt = text(
        "INSERT INTO conversation_memory (conversation_id, embedding, text) "
        "VALUES (:cid, :emb, :txt)"
    )
    with engine.connect() as conn:
        conn.execute(
            insert_stmt,
            {"cid": conversation_id, "emb": embedding, "txt": answer_text},
        )
        conn.commit()
@app.post("/query/")
def query(req: QueryRequest):
    # 1️⃣ Retrieve relevant documents
    docs = vectorstore.similarity_search(req.question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # 2️⃣ Fetch previous conversation memory
    memory_context = fetch_conversation_memory(req.conversation_id)

    # 3️⃣ Build prompt
    prompt = f"""
You are an AI assistant. Use the context below to answer the question.

Context from documents:
{context_text}

Previous conversation memory:
{memory_context}

Question:
{req.question}

Answer concisely and provide sources.
"""

    # 4️⃣ Call Mistral API
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MISTRAL_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    print("Mistral status:", response.status_code)
    print("Mistral response:", response.text)

    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Mistral API error: {response.text}")

    # 5️⃣ Extract answer
    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid Mistral response format: {e}")

    # 6️⃣ Embed and store memory
    embedding_vector = embeddings_model.embed_query(answer)
    store_conversation_memory(req.conversation_id, answer, embedding_vector)

    # 7️⃣ Return answer + sources
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    return {"answer": answer, "sources": sources}

