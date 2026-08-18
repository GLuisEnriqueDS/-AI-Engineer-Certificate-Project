import os
from dotenv import load_dotenv
from groq import Groq

# Cargar variables de entorno
load_dotenv()

# Crear cliente
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Hacer una consulta
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "¿Cuándo fue la Segunda Guerra Mundial?"
        }
    ]
)

print(response.choices[0].message.content)