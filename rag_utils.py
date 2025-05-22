import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Charger les variables d'environnement
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # depuis ton .env
genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data"
VECTOR_DIR = "vector_store"

def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)
    return docs

def chunk_documents(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for text in texts:
        chunks = splitter.create_documents([text])
        docs.extend(chunks)
    return docs

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_DIR)
    return db

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return db, retriever

def generate_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur Gemini API : {e}"

# def get_answer(query, retriever):
#     docs = retriever.get_relevant_documents(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     prompt = f"Contexte:\n{context}\n\nQuestion: {query}\nRéponse:"
#     return generate_response(prompt)

def get_answer(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

  
    instruction = (
    "Tu es Felah, un assistant agricole expert et amical.\n"
    "Avant de répondre, analyse l intention de l utilisateur :"
    "- Est-ce une salutation ? une remerciement ? une vraie question ?\n"
    "Adapte ta réponse en conséquence."
    "Réponds toujours en meme langue de quastion , si la question en francais répond en francais et si la question en arabe répond en arabe, de façon claire et adaptée à un agriculteur.\n"
    "Fais un retour à la ligne avant de répondre à la question.\n"
    "Si l'utilisateur te dit seulement 'bonjour', 'merci', ou te salue, réponds simplement avec une courte formule polie adaptée (ex : 'Bonjour ! Comment puis-je vous aider ?').\n"
    "Si l'utilisateur pose une vraie question, donne une réponse structurée, simple et pédagogique.\n"
    "il faut dire bonjour à chaque fois tu répond , seulement une fois au début de discussion , et après utilise d'autre formule de politesse"
    "Ne donne jamais de réponses inutiles ou trop longues pour des messages courts.\n"
    "Sois clair, structuré, sans jargon technique inutile.\n"
    "Si tu ne sais pas répondre précisément, dis-le honnêtement.\n"
    "Termine toujours ta réponse par une invitation à poser une autre question, sauf si c'est une salutation.\n"
     )


    prompt = (
        f"{instruction}\n\n"
        f"Contexte :\n{context}\n\n"
        f"Question : {query}\n"
        f"Réponse :"
    )

    return generate_response(prompt)
