

# FastAPI RAG LLM Assistant

A **Retrieval-Augmented Generation (RAG)** system built with **FastAPI** and **Mistral LLM**. This project allows you to ask questions about your documents and get AI-generated answers, while also storing conversation memory for context-aware responses.

## 🚀 Project Overview

This project combines:

- **FastAPI**: A modern, fast web framework for building APIs.
- **LangChain / HuggingFace Embeddings**: To create vector embeddings of your documents.
- **PostgreSQL + PGVector**: Stores embeddings and conversation memory for retrieval.
- **Mistral API (LLM)**: Generates answers based on retrieved documents and previous conversations.

It implements a **RAG pipeline**:

1. User sends a question.
2. Relevant document chunks are retrieved from the vector store.
3. Previous conversation memory is fetched.
4. A prompt is built including document context + conversation memory.
5. The LLM (Mistral) generates an answer.
6. The answer is embedded and stored in the database for future context.
7. Answer and sources are returned to the user.

---

## ⚙️ How It Works

```

User Question
│
▼
Retrieve relevant docs from vector store
│
▼
Fetch conversation memory from database
│
▼
Build prompt for LLM
│
▼
Call Mistral API → Generate answer
│
▼
Store answer embedding in DB for future memory
│
▼
Return answer + sources to user

```

---

## 📂 Project Structure

```

rag/
├─ app/
│  ├─ main.py         # FastAPI API with RAG logic
│  ├─ database.py     # DB connection & vectorstore initialization
├─ data/
│  ├─ pdfs/           # Your PDF documents to query
├─ .env               # Store environment variables (MISTRAL_API_KEY, DB credentials)
├─ requirements.txt   # Python dependencies
├─ README.md

````

---

## 🛠️ Setup & Installation

1. **Clone the repo**

```bash
git clone https://github.com/<your-username>/fastapi-rag-llm.git
cd fastapi-rag-llm
````

2. **Create virtual environment and install dependencies**

```bash
python -m venv rag-env
source rag-env/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables** in `.env` file:

```
MISTRAL_API_KEY=your_mistral_api_key
DB_USER=rag_user
DB_PASSWORD=your_password
DB_NAME=rag_db
DB_HOST=localhost
DB_PORT=5432
```

4. **Initialize your database and vectorstore** (make sure `rag_db` exists):

```bash
psql -U postgres -c "CREATE DATABASE rag_db;"
# Run any DB migrations if required
```

5. **Run the API server**

```bash
uvicorn app.main:app --reload
```

Server will run at: `http://127.0.0.1:8000`

---

## 💬 How to Use

Send a POST request to `/query/` with JSON payload:

```bash
curl -X POST "http://127.0.0.1:8000/query/" \
-H "Content-Type: application/json" \
-d '{
  "question": "Explain Supabase Authentication",
  "conversation_id": "test1"
}'
```

**Response:**

```json
{
  "answer": "Supabase Authentication simplifies user management...",
  "sources": ["data/pdfs/Authentication_Methods.pdf"]
}
```

---

## 🔧 Integrating into Your Project

You can integrate this API into your project by:

1. Making POST requests to `/query/` from your frontend or backend.
2. Storing conversation IDs to maintain context across sessions.
3. Adding more documents to your vector store to expand knowledge base.

> **Tip:** Use `conversation_id` per user session to maintain personalized memory.

---

## 📌 Notes

* Only **FastAPI backend** is included. Streamlit or other frontends can be added separately.
* Uses **CPU embeddings** by default (`all-MiniLM-L6-v2`), so no GPU is required.
* Database memory stores both embeddings and raw text to provide context for RAG.

---

## ⚡ License

MIT License

