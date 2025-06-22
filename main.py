from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uvicorn

from rag_utils import get_answer
from image_predictor import predict_plant_disease

app = FastAPI()

# ✔️ Route d'accueil (évite l'erreur 404 sur Render)
@app.get("/")
async def root():
    return {"message": "✅ Felah est opérationnel"}

# 📩 Schéma pour la question texte
class Question(BaseModel):
    query: str

# 🔍 Endpoint pour poser une question texte
@app.post("/ask")
async def ask_question(q: Question):
    response = get_answer(q.query)
    return JSONResponse(content={"answer": response})

# 🌿 Endpoint pour prédiction d'image de plante
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    return await predict_plant_disease(file)

# 🖥️ Lancement local
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
