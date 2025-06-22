from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uvicorn
from rag_utils import get_answer


from image_predictor import predict_plant_disease




app = FastAPI()

# Initialiser le retriever
# vector_store, retriever = load_vector_store()

# Schéma de question texte
class Question(BaseModel):
    query: str

# Endpoint pour chatbot texte

@app.post("/ask")
async def ask_question(q: Question):
    response = get_answer(q.query)
    return JSONResponse(content={"answer": response})



@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    return await predict_plant_disease(file)


# Run local
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
