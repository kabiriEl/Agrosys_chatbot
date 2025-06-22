
import os
import fitz  # PyMuPDF
from rag_utils import index_documents

DATA_FOLDER = "data"

def extract_text_chunks_from_pdf(pdf_path, max_chunk_len=1000):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Découpage du texte en chunks
    chunks = []
    for i in range(0, len(full_text), max_chunk_len):
        chunk = full_text[i:i + max_chunk_len]
        if len(chunk.strip()) > 100:  # Ignore les petits bouts
            chunks.append(chunk.strip())
    return chunks

def build_all_chunks(data_dir):
    all_chunks = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            chunks = extract_text_chunks_from_pdf(path)
            print(f"{filename} → {len(chunks)} chunks")
            all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    print("⏳ Extraction et vectorisation des fichiers PDF...")
    all_chunks = build_all_chunks(DATA_FOLDER)
    index_documents(all_chunks)
    print(f"✅ {len(all_chunks)} chunks indexés dans Pinecone.")












# from rag_utils import load_documents, chunk_documents, create_vector_store

# if __name__ == "__main__":
#     print("🔍 Chargement des documents...")
#     texts = load_documents()
#     print(f"Nombre de documents : {len(texts)}")
#     print("✂️ Découpage des documents...")
#     chunks = chunk_documents(texts)
#     print(f"Nombre de chunks : {len(chunks)}")
#     print("📦 Création de l'index FAISS...")
#     create_vector_store(chunks)


#     print("✅ Index FAISS créé et sauvegardé dans /vector_store")

