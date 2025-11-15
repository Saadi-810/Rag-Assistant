import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from db.database import get_vectorstore

# ---------- CONFIG ----------
PDF_FOLDER = "/home/saadi/Documents/rag/data/pdfs"  # Folder where your PDFs are stored
CHUNK_SIZE = 800           # Chunk size in characters
CHUNK_OVERLAP = 100        # Overlap to preserve context continuity
# ----------------------------

def ingest_pdfs():
    vectorstore = get_vectorstore()

    # Get all PDFs from the folder
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    all_docs = []
    for pdf in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Chunking strategy explanation:
    # RecursiveCharacterTextSplitter keeps semantic meaning while splitting long code blocks or paragraphs.
    # Chunk size 800 chars is ideal for one-pager PDFs (balance between detail and embedding cost).

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)

    # Store embeddings + metadata in vector store
    vectorstore.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks from {len(pdf_files)} PDFs into vector DB.")


if __name__ == "__main__":
    ingest_pdfs()
