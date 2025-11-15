from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings



# Database connection URL
DATABASE_URL = "postgresql+psycopg2://rag_user:rag_password@localhost/rag_db"

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Embedding model (you can switch to local later)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
# Vector store setup
COLLECTION_NAME = "rag_documents"

def get_vectorstore():
    vectorstore = PGVector(
        connection_string=DATABASE_URL,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return vectorstore
