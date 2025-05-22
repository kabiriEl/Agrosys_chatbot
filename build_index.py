from rag_utils import load_documents, chunk_documents, create_vector_store

if __name__ == "__main__":
    print("🔍 Chargement des documents...")
    texts = load_documents()
    print(f"Nombre de documents : {len(texts)}")
    print("✂️ Découpage des documents...")
    chunks = chunk_documents(texts)
    print(f"Nombre de chunks : {len(chunks)}")
    print("📦 Création de l'index FAISS...")
    create_vector_store(chunks)


    print("✅ Index FAISS créé et sauvegardé dans /vector_store")
