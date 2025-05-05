# EVALIA - módulo para evaluación automática mediante LLM

Módulo Python para asistir en evaluación por IA, apoyada en la API de OpenAI (GPT).

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
* `GEMINI_API_KEY`. Para poder utilizar la API de Gemini (Google).
* `GOOGLE_CLOUD_PROJECT`: Id del proyecto de Google Cloud, necesario para poder usar el procesamiento por lotes de Gemini.
* 

* `EVALIA_CACHE_DIR` Ruta de los archivos temporales de la aplicación (archivos _pickle_). Si se omite, se usa la ruta por defecto para los archivos de caché de "evalia".
* `EVALIA_LOG_DIR` Ruta de los archivos de registro (_logs_) de la aplicación. Si se omite, se usa la ruta por defecto del sistema operativo para los _logs_ de "evalia".
* `EVALIA_RUN_HIGH_COST_TESTS` (para el desarrollador). Si vale 1, habilita la ejecución de unidades de test de alto coste potencial (que consumen mucho tiempo o que interactúan mucho con el motor de IA).

## Ejemplo básico

Pueden verse varios ejemplos en la carpeta [examples](examples). 
A continuación se muestra el ejemplo inicial, [example01.py](examples/example01.py).

```python
import pandas as pd

# Clase para evaluador automático
from evalia.evaluators import Evaluator

# Para leer prompts desde cadenas de texto
from evalia.prompts import PromptFromString

# Para indicar el modelo GPT que se va a usar
MODELO_GPT = 'gpt-4o-mini'

# Un prompt
mi_prompt = '''
Eres un evaluador de geografía y te han pedido que evalúes si los estudiantes 
conocen las capitales de los países europeos.
Al estudiante se le da la siguiente instrucción:
"escribe una lista de cinco capitales europeas".
A continuación te pasaré una lista de respuestas de estudiantes. 
Cada respuesta tiene un número de índice y la lista de ciudades.
Tienes que calificar cada respuesta de la siguiente forma:
1 = hay al menos cinco nombres y todos son capitales europeas.
0 = cualquier otro caso.
No importan las faltas de ortografía: 
por ejemplo, considera correctas "Berlin" y "Verlin".
Tu calificación debe venir en este formato: <número de respuesta>. <calificación>
'''

# Un conjunto de respuestas para evaluar
respuestas_estudiantes = {
    "respuesta": [
        "Madrid, París, Berlín, Roma, Lisboa",
        "Madriz, Paris, Verlin, Rroma, Lisbona",  # con faltas de ortografía
        "París, Londres",
        "Pekín, Tokio, París, Roma, Copenhague, Londres",
        "esto es una respuesta inválida",
        "Madrid, París, Berlín, Roma, Lisboa, Londres",
        "Barcelona, Londres, Rotterdam, Salzburgo"
    ],
    "calificación real": [1, 1, 0, 0, 0, 1, 0]
}

# El evaluador automático
evaluador = Evaluator(
    evaluator_id="capitales europeas",
    student_responses=pd.DataFrame(respuestas_estudiantes),
    responses_column="respuesta",
    system_context=PromptFromString(mi_prompt),
    gpt_manager=MODELO_GPT,
    query_batch_length=20
)

# El parámetro query_batch_length indica cómo se empaquetan las peticiones a GPT.
# El valor 20 significa que las respuestas se enviarán a GPT en grupos de 20.

# Ejecuta la evaluación y devuelve un dataframe con el resultado
df_result = evaluador.run()

print("Resultados:")
print(df_result)

# Imprime estadísticas: tokens y tiempo consumido
print(evaluador.print_stats())
```

## Comentarios al ejemplo

### Diseño del _prompt_

Internamente, el framework usa un número de respuesta para asociar 
las respuestas de GPT con los datos de origen. Es importante indicarlo a GPT:

```
Cada respuesta tiene un número de índice y la lista de ciudades.
```

También es importante indicarle a GPT cómo debe dar formato a
sus evaluaciones. En este caso, le decimos que devuelva un número de respuesta 
seguido de un punto y luego el texto de la evaluación:

```
Tu calificación debe venir en este formato: <número de respuesta>. <calificación>
```

Por defecto, en este framework se espera que el resultado de GPT sea de una única
línea por cada respuesta evaluada. 
El sistema está preparado para procesar resultados más complejos y flexibles, 
por ejemplo que GPT devuelva objetos JSON. 
Esto se controla mediante la clase `GPTResponse` y sus herederas.

En el ejemplo [example02.py](examples/example02.py) se muestra cómo trabajar 
con resultados en formato JSON.

### DataFrame con el resultado 

El _data frame_ con el resultado tiene la misma estructura que el original, 
con dos columnas añadidas:

- "Calificación GPT"
- "Respuesta completa GPT"

Por defecto, la columna "Respuesta completa GPT" tiene el texto de GPT, 
sin procesar. La columna "Calificación GPT" tiene el texto, quitándole
el número de índice de la respuesta.

El programador puede sobreescribir el método
`Evaluator.postprocess_one_gpt_response()` para procesar la respuesta
de GPT y obtener una calificación sencilla, que irá a 
la columna "Calificación GPT". 
El ejemplo [ejemplo03.py](examples/example03.py) tiene una muestra de cómo
hacer ese tratamiento.

### Agrupar las peticiones en lotes (query_batch_length)

El parámetro `query_batch_length` indica cómo se empaquetan las peticiones a GPT. 
El valor 20 del ejemplo significa que las respuestas 
se enviarán a GPT en grupos de 20.
El agrupamiento en lotes ayuda a reducir costes de uso del GPT, ya que todo el
lote comparte un único _prompt_ de instrucciones.

El tamaño del lote no afecta a la estructura o al contenido del _data frame_
de respuesta.


## Código fuente: Ficheros principales

- __[evaluators.py](src/evalia/evaluators.py)__. Clases para implementar la evaluación de los ítems. Todas las evaluaciones implementan la interfaz de la clase abstracta `AbstractEvaluator`. La clase base concreta `Evaluator` contiene una implementación totalmente funcional de todas las operaciones.
- __[gpt_manager/gpt_manager.py](src/evalia/llm/gpt/gpt_manager.py)__. Clase abstracta `GPTManager`. 
Una interfaz sencilla con la API de OpenAI, adaptada a nuestro sistema. 
Implementa contención automática del tráfico con OpenAI,
para evitar superar los límites de tokens por minuto y de peticiones por minuto.
- __[gpt_responses.py](src/evalia/evaluated_answer.py)__. Clases para el tratamiento de las evaluaciones procedentes de GPT. Se definen tres clases concretas: `GPTResponseOneLine`, `GPTResponseMultiline` y `GPTResponseJSON`, según si las respuestas vienen en una línea, en bloques de texto o en una lista JSON.
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

