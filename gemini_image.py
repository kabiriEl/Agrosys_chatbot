import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os

# Initialise Gemini multimodal
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_response(prompt: str, image: Image.Image = None) -> str:
    try:
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"
