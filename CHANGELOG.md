# Changelog
Todas las modificaciones importantes de este proyecto quedan documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/) y sigue [Semantic Versioning](https://semver.org/).

---
## [0.4.0] - 2026-09-28
### Añadido
- `Evaluator.create_or_resume(evaluator_id, factory=None)`: 
  Recupera el evaluador guardado previamente con ese id (archivo pickle).
  Si no hay ninguno guardado, crea uno nuevo con `factory(evaluator_id)`. 
  Si se omite _factory_ se usa el constructor de la propia clase, que debe soportar
  construirse solo con un parámetro "id". 
  Si se reanuda desde pickle no se ejecutan ni el constructor ni la _factory_, y se avisa por consola.
- `Evaluator.discard_saved(evaluator_id)`: borra el evaluador guardado (el archivo pickle).
- Atributo de clase `GPTManager.requires_persistence` (verdadero en `GPTBatchManager`): indica si el
  evaluador debe persistir para poder reanudar la tarea más tarde.

### Cambiado
- El evaluador solo se guarda en disco si su manager lo necesita (Batch API). Antes se guardaba siempre,
  también en modo síncrono: un programa con `@Evaluator.persistent` que se volvía a ejecutar recuperaba
  las respuestas ya obtenidas. Ahora, en modo síncrono, la persistencia no tiene ningún efecto.
- Errores claros al crear o reanudar un evaluador guardado: si es de otra clase (p. ej. dos evaluaciones
  con el mismo id), si el evaluador creado tiene otro id o si el fichero guardado no se puede leer.

### Obsoleto
- `@Evaluator.persistent`: usar `Evaluator.create_or_resume(id, factory)`. Sigue funcionando, con un
  aviso `DeprecationWarning` y las mismas comprobaciones que `create_or_resume()`.


## [0.3.0] - 2026-09-28
### Añadido
- `GPTSmartManager` (modo síncrono) funciona con cualquier modelo de OpenAI, incluidos gpt-5.x y gpt-6.
  Antes solo admitía los modelos de una tabla interna de límites: gpt-3.5-turbo, gpt-4, gpt-4-turbo,
  gpt-4o y gpt-4o-mini.
- `accepts_temperature(model)` en `evalia.gpt_manager.gpt_manager`: indica si un modelo admite una
  temperatura distinta de la de por defecto.

### Cambiado
- Los límites RPM y TPM de la cuenta se descubren automáticamente al crear un `GPTSmartManager`, con una
  petición mínima al modelo (cabeceras `x-ratelimit-*`). Por tanto, crear el manager requiere red y una
  clave válida, desde el mismo instante de la construcción del objeto.
- La temperatura solo se envía a los modelos que la admiten (gpt-3.5, gpt-4.x y gpt-5.1). Los modelos de
  razonamiento (gpt-5, gpt-5.5, gpt-6…) usan su valor por defecto (1), porque la API rechaza cualquier
  otro. Es una solución provisional, en espera de un cambio en la API de temperatura.

### Eliminado
- La variable de entorno `OPENAI_TIER` ya no es necesaria. Si está definida, se ignora.

### Corregido
- Estimación de tokens: gpt-4o, gpt-4.1 y gpt-4.5 usaban el tokenizador `cl100k_base` en lugar de
  `o200k_base`. Los modelos que `tiktoken` no conoce (p. ej. gpt-6) usan ahora `o200k_base` en lugar
  de provocar un error.
- `GPTSmartManager.query()` aborta si se produce un error no recuperable mediante reintentos.
  Las versiones anteriores reintentaban en todas las situaciones, por ejemplo si había un error de 
  autenticación. Ahora solo se reintenta ante estos problemas: APITimeoutError, APIConnectionError, RateLimitError.

## [0.2.0] - 2025-03-31
### Añadido
- Se parametrizan los directorios de salida de la aplicación, mediante variables de entorno y, por defecto, mediante los directorios del sistema (módulo platformdirs).

## [0.1.0] - 2025-03-13
### Añadido
- Primera versión funcional subida a GitHub.

