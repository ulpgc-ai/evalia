import os
import openai
import time

openai.api_key = os.getenv('OPENAI_API_KEY')

# Obtiene la lista de modelos disponibles
available_models = openai.models.list()
print([model.id for model in available_models.data])

start_time = time.time()
response = openai.chat.completions.create(
#    model = "gpt-3.5-turbo",
    model = "gpt-4",
    messages = [
        {"role": "system", "content" : "Eres un asistente mentiroso, siempre das respuestas verosímiles pero erróneas"},
 #       {"role": "system", "content" : "You are a liar assistant, you always emit plausible but wrong answers"},
        { "role" : "user", "content" : "¿Cuál es la capital de España?" },
        { "role" : "assistant", "content" : "Barcelona" },
        { "role" : "user", "content" : "¿Cuál es la capital de Francia?" },
        { "role" : "assistant", "content" : "Marsella" },
        { "role" : "user", "content" : "¿Cuál es la capital de Zimbabwe?" }
    ],
    temperature = 0.4,
    max_tokens = 1024
)
end_time = time.time()

print(f"elapsed time (secs): {end_time-start_time}")
print("tokens spent: ")
print(response.usage)

print("Answer: ")
print(response.choices[0].message.content)
