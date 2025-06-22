from PIL import Image
from io import BytesIO
from fastapi import UploadFile
from gemini_image import generate_response  

async def predict_plant_disease(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        prompt = (
            "Tu es un expert en santé des plantes. Observe cette image de plante et fais un diagnostic.\n"
            "Réponds en respectant scrupuleusement ce format :\n\n"
            "1. Démarre par : 'D’après l’image'\n"
            "2. Donne le nom probable de la maladie et une explication scientifique courte\n"
            "3. Propose 2 à 3 conseils simples et pratiques pour l’agriculteur\n"
            "4. Termine par : 'N’hésitez pas à poser une autre question.'\n\n"
            "Utilise un ton professionnel et accessible. Ne réponds pas si la plante est trop floue ou si tu ne peux pas diagnostiquer."
        )

        # Envoie l'image + prompt à Gemini
        explanation = generate_response(image=image, prompt=prompt)

        return {
            "réponse": explanation
        }

    except Exception as e:
        return {"erreur": str(e)}
