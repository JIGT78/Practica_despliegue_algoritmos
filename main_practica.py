from fastapi import FastAPI, HTTPException
import os
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline
import math

app = FastAPI()

# 1.Saludo
@app.get('/saluda')
def saluda(nombre: str, apellido: str,):
    return {'Message': f'Hola {nombre} {apellido}.'}


# 2.Calculo volumen cilindro
@app.get('/volumen_cilindro')
def volumen_cilindro(radio: float, altura: float):
    if radio <= 0 or altura <= 0:
        raise HTTPException(
            status_code=400,
            detail="El radio y la altura deben ser mayores que 0."
        )
    volumen = math.pi * (radio ** 2) * altura
    return {"Volumen": volumen}


# 3️.Análisis de sentimiento (HF)
@app.get('/sentiment')
def sentiment_classification(prompt):
  sentiment_pipeline = pipeline('sentiment-analysis')
  return {'Sentiment': sentiment_pipeline(prompt)[0]['label']}

# 4.Resumen de texto (HF)
@app.get('/resumen')
def resumen_texto(text: str, max_length: int = 50, modelo: str = "facebook/bart-large-cnn"):
    # Cargar el pipeline de resumen con el modelo especificado
    summarizer = pipeline("summarization", model=modelo)
    
    # Obtener el resumen del texto
    resultado = summarizer(text, max_length=max_length, min_length=10, do_sample=False)
    
    # Retornar el resumen
    return {'Resumen': resultado[0]['summary_text']}

# 5. Traducción español-inglés (HF)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

@app.get('/traducir')
def traducir(texto: str):
    try:
        resultado = translator(texto)[0]['translation_text']
        return {'Traducción': resultado}
    except Exception as e:
        return {'error': str(e)}

