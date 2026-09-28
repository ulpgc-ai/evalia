# Changelog
Todas las modificaciones importantes de este proyecto quedan documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/) y sigue [Semantic Versioning](https://semver.org/).

---

## [Sin publicar]
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

