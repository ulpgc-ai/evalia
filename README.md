# EVALIA - módulo para evaluación automática mediante LLM

Módulo Python para asistir en evaluación por IA, apoyada en la API de OpenAI (GPT), Google (Gemini) y Anthropic (Claude).

---
© Universidad de Las Palmas de Gran Canaria, 2023-2025. Todos los derechos reservados.

## Configuración

### Instalación

El módulo está preparado para instalarse con PIP. 

El archivo [pyproject.toml](pyproject.toml) indica las dependencias de este módulo.

### Variables de entorno

#### OpenAI
* `OPENAI_API_KEY`. Para poder utilizar la API de OpenAI.
* `OPENAI_TIER`. Opcional, para indicar en qué nivel de contrato está la cuenta de OpenAI. 
Se usará para controlar los límites RPM y TPM de la interacción con GPT. 
Si la variable no está definida, se usará "Tier 1".
Otros valores pueden ser "Tier 2", "Tier 3", "Tier 4" y "Tier 5".

#### Gemini
* `GOOGLE_API_KEY`. Para poder utilizar la API de Gemini (Google).
* `GOOGLE_CLOUD_PROJECT`: Id del proyecto de Google Cloud. Por ejemplo, "gen-lang-client-123456".
* `GOOGLE_CLOUD_LOCATION`: Ubicación del proyecto de Google Cloud. Por ejemplo, "europe-west1" o "us-central1".
* `GOOGLE_CLOUD_STORAGE_BUCKET_NAME`: Nombre del bucket de Google Cloud Storage donde se guardarán los ficheros temporales de procesamiento por lotes de Gemini. Por ejemplo, "evalia-test".
* `GOOGLE_CLOUD_STORAGE_JSONL_DESTINATION_URI`: El directorio dentro del bucket en el que se guardará el JSONL de entrada.
* `GOOGLE_CLOUD_STORAGE_BATCH_OUTPUT_URI`: El directorio dentro del bucket en el que se guardará el JSONL de salida.

#### Claude
* `ANTHROPIC_API_KEY`. Para poder utilizar la API de Anthropic (Claude).

#### Evalia
* `EVALIA_CACHE_DIR` Ruta de los archivos temporales de la aplicación (archivos _pickle_). Si se omite, se usa la ruta por defecto para los archivos de caché de "evalia".
* `EVALIA_LOG_DIR` Ruta de los archivos de registro (_logs_) de la aplicación. Si se omite, se usa la ruta por defecto del sistema operativo para los _logs_ de "evalia".
* `EVALIA_RUN_HIGH_COST_TESTS` (para el desarrollador). Si vale 1, habilita la ejecución de unidades de test de alto coste potencial (que consumen mucho tiempo o que interactúan mucho con el motor de IA).

## Ejemplo básico

Pueden verse varios ejemplos en la carpeta [examples](examples). 
A continuación se muestra el ejemplo inicial, [example01.py](examples/example01.py).

```python
'''
EJEMPLO 1. Evaluar capitales europeas
'''

import pandas as pd
import os

# Clase para evaluador automático
from evalia.evaluator import Evaluator
from evalia.llm.evaluated_answer import EvaluatedAnswers
from evalia.llm.gemini import GeminiBasicManager
from evalia.llm.claude import ClaudeBasicManager
from evalia.llm.gpt import GPTSmartManager

# Para leer prompts desde cadenas de texto
from evalia.prompts import PromptFromString

# Para configurar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4o-mini'

# Directorio de salida
OUTPUT_DIR = "./output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROMPT = '''
Eres un evaluador de geografía y te han pedido que evalúes si los estudiantes 
conocen las capitales de los países europeos.
Al estudiante se le da la siguiente instrucción:
"escribe una lista de cinco capitales europeas".
A continuación te pasaré una lista de respuestas de estudiantes. 
Cada respuesta tiene un número de índice y la lista de ciudades.
Tienes que calificar cada respuesta de la siguiente forma:
1 = hay al menos cinco nombres y todos son capitales europeas.
0 = cualquier otro caso.
No importan las faltas de ortografía, por ejemplo considera correctas "Berlin" y "Verlin".
'''

respuestas_estudiantes = {
    "respuesta": [
        "Madrid, París, Berlín, Roma, Lisboa",
        "Madriz, Paris, Verlin, Rroma, Lisbona", # con faltas de ortografía
        "París, Londres",
        "Pekín, Tokio, París, Roma, Copenhague, Londres",
        "esto es una respuesta inválida",
        "Madrid, París, Berlín, Roma, Lisboa, Londres",
        "Barcelona, Londres, Rotterdam, Salzburgo"
        ],
    "calificación real": [1, 1, 0, 0, 0, 1, 0]
}


evaluador = Evaluator(
    evaluator_id = "capitales europeas",
    answers_dataframe= pd.DataFrame(respuestas_estudiantes),
    prompt= PromptFromString(PROMPT),
    query_batch_length=2
)
evaluador.add_manager(GPTSmartManager(model="gpt-4o-mini"))
evaluador.add_manager(GeminiBasicManager(model="gemini-2.0-flash"))
evaluador.add_manager(ClaudeBasicManager(model="claude-3-5-sonnet-latest"))

# Ejecuta la evaluación y devuelve un dataframe con el resultado
df_result = evaluador.evaluate_answers()

print("Resultados:")
print(df_result)

# Imprime estadísticas: tokens y tiempo consumido
evaluador.print_stats()

# Guarda el resultado en un Excel
EXCEL_FILE = os.path.join(OUTPUT_DIR, "example01.xlsx")
df_result.to_excel(EXCEL_FILE)
```

## Comentarios al ejemplo

### Diseño del _prompt_

Internamente, el framework usa un número de respuesta para asociar las respuestas del LLM con los datos de origen. Es importante indicarlo al LLM:

```
Cada respuesta tiene un número de índice y la lista de ciudades.
```

Por defecto, en este framework se espera que el resultado de cada LLM sea un JSON de tipo `EvaluatedJustifiedAnswers.` (parámetro `structured_output_class` de cada `LanguageModelManager`). Este objeto tiene un parámetro `results`, que es una lista de objetos de tipo `EvaluatedJustifiedAnswer` (que contiene un índice, una calificación y una justificación).

### DataFrame con el resultado 

El _data frame_ con el resultado tiene la misma estructura que el original, 
con dos columnas añadidas:

- "Calificación (nombre del LLM)"
- "Respuesta completa (nombre del LLM)"

La columna "Respuesta completa (nombre del LLM)" contiene la justificación de la calificación si se ha escogido `EvaluatedJustifiedAnswers` como clase de respuesta estructurada (parámetro `structured_output_class` de cada `LanguageModelManager`). Si no se ha escogido, esta columna no se añadirá.

### Agrupar las peticiones en lotes (query_batch_length)

El parámetro `query_batch_length` indica cómo se empaquetan las peticiones a cada LLM. 
El valor 20 del ejemplo significa que las respuestas 
se enviarán a cada LLM en grupos de 20.
El agrupamiento en lotes ayuda a reducir costes de uso de los LLM, ya que todo el
lote comparte un único _prompt_ de instrucciones.

El tamaño del lote no afecta a la estructura o al contenido del _data frame_
de respuesta.


## Código fuente: Ficheros principales

- __[evaluator.py](src/evalia/evaluator.py)__. Clase principal de ``EVALIA``. Evalúa un conjunto de respuestas de estudiantes, usando un _prompt_ y un conjunto de modelos de lenguaje (LLM) para la evaluación.
- __[llm/gpt/gpt_manager.py](src/evalia/llm/gpt/gpt_manager.py)__. Clase abstracta para interactuar con diferentes modelos de lenguaje de OpenAI.
    Esta clase hereda de LanguageModelManager y proporciona una interfaz para interactuar
    con modelos de lenguaje específicos de OpenAI.
- __[llm/gemini/gemini_manager.py](src/evalia/llm/gemini/gemini_manager.py)__. Clase abstracta para interactuar con diferentes modelos de lenguaje de Google Gemini.
    Esta clase hereda de LanguageModelManager y proporciona una interfaz para interactuar
    con modelos de lenguaje específicos de Google.
- __[llm/claude/claude_manager.py](src/evalia/llm/claude/claude_manager.py)__. Clase abstracta para interactuar con diferentes modelos de lenguaje de Anthropic Claude.
    Esta clase hereda de LanguageModelManager y proporciona una interfaz para interactuar
    con modelos de lenguaje específicos de Anthropic.
- __[prompts.py](src/evalia/prompts.py)__. Clases que producen instrucciones (_prompts_) a partir de distintas fuentes: fichero de texto plano, fichero JSON, etc.

## Arquitectura del software

En el documento [class_architecture.md](class_architecture.md) se describe el diseño de clases Python de esta biblioteca _(documento pendiente de revisión)_.


## Ficheros que genera la ejecución de EVALIA

Cuando se ejecuta Evalia o un programa de prueba, esta biblioteca puede generar estos ficheros:

- __Ficheros pickle__ Se usan para persistir los objetos cuando se ejecutan tareas 
en modo lote (_batch_). Se almacenan en el directorio definido en la variable de entorno
__EVALIA_CACHE_DIR__ o, en su defecto, en la ruta de cachés de la aplicación "evalia", 
según el estándar de la máquina donde se ejecuta el módulo.
- __app.log__ Archivo de registro (_log_) de actividad de la aplicación. 
Se guarda en el directorio definido en la variable de entorno __EVALIA_LOG_DIR__ o, en su defecto, en la ruta de _logs_ de la aplicación "evalia", según el estándar
de la máquina en la que se ejecuta el módulo (ej. C:\App Data\evalia\logs).

## Documentación

Para generar la documentación de este módulo, se ha usado [Sphinx](https://www.sphinx-doc.org/en/master/). En la carpeta `docs` se encuentra el código fuente de la documentación, que se puede generar en formato HTML y PDF.