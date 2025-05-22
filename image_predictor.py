from transformers import pipeline
from PIL import Image
from io import BytesIO
from fastapi import UploadFile
from gemini_utils import generate_response

# Charger le modèle une fois
pipe = pipeline("image-classification", model="swueste/plant-health-image-classifier")

async def predict_plant_disease(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        predictions = pipe(image)

        # On prend uniquement la prédiction la plus probable
        top_pred = predictions[0]
        label = top_pred["label"]
        score = round(top_pred["score"] * 100, 2)

        prompt = (
        f"D'après l'image, la plante semble atteinte de '{label}' avec une probabilité de {score}%. "
        "Tu es un expert en agriculture. Reformule ce diagnostic de manière claire, professionnelle et courte, "
        "en respectant scrupuleusement le format suivant :\n\n"
        "1. Démarre par : 'D’après l’image'\n"
        "2. Donne le nom de la maladie et une brève explication scientifique (1 à 2 lignes max)\n"
        "3. Propose 2 à 3 conseils simples et pratiques pour l’agriculteur\n"
        "4. Termine toujours par : 'N’hésitez pas à poser une autre question.'\n\n"
        "Utilise un langage accessible mais professionnel. N’ajoute pas d’informations non nécessaires."
         )



        explanation = generate_response(prompt)

        return {
            "réponse": explanation
        }

    except Exception as e:
        return {"erreur": str(e)}
