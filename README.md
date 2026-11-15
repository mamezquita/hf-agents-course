# Hugging Face AI Agents Course

Repositorio personal del **AI Agents Course** de Hugging Face
(https://huggingface.co/learn/agents-course), realizado como complemento
del examen final de Deep Learning 2026-2 – Universidad Externado de Colombia.

- **Estudiante:** Miguel Ángel Amézquita Briceño
- **Username de Hugging Face:** [MiguelAngelAB](https://huggingface.co/MiguelAngelAB)
- **Certificado obtenido:** Fundamentals (Unidad 1 completa)

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `unit1/dummy_agent_library.ipynb` | Agente construido "a mano": system prompt con herramientas, ciclo Thought → Action → Observation, `stop` y función dummy `get_weather` |
| `unit1/tools.ipynb` | Clase `Tool`, decorador `@tool` y prueba de una herramienta (calculadora) con un LLM real |
| `unit1/first_agent_smolagents.ipynb` | Primer agente con smolagents (`CodeAgent`) con herramientas de hora, días restantes y búsqueda web |

Todos los notebooks están ejecutados en Google Colab y tienen las salidas visibles.
Como LLM se usa **Gemini** (`gemini-4-hydrogen`) con una API key gratuita de Google AI Studio,
en lugar de la API de Hugging Face, que requiere créditos de pago. La llave se guarda en los
*Secrets* de Colab y no aparece en el código.

## Observaciones

- **Unidad 0:** onboarding completo (cuenta de HF, Discord y seguimiento a la organización agents-course).
- **Dummy Agent Library:** se reemplazó el cliente de Hugging Face por el endpoint de Gemini compatible
  con OpenAI. A diferencia de lo que muestra el curso, Gemini no alucinó la *Observation* en la primera
  llamada; aun así el `stop=["Observation:"]` es necesario porque no todos los modelos se comportan igual.
  Gemini no acepta conversaciones que terminen con un mensaje del asistente (*prefill*), así que se agregó
  un mensaje de usuario pidiendo continuar desde la *Observation*.
- **Tools:** se comprobó que la descripción generada automáticamente con el decorador `@tool` es idéntica
  a la escrita a mano, y que el LLM puede usar la herramienta: pidió la calculadora, el código la ejecutó
  y respondió correctamente 1234 × 5678 = 7006652.
- **Primer agente con smolagents:** el curso pide duplicar el Space `First_agent_template`, pero Hugging
  Face ya no permite crear Spaces de Gradio con cuenta gratuita (*CPU Basic* y *ZeroGPU* requieren PRO).
  Por eso el agente se implementó en Colab con el mismo framework, el mismo `CodeAgent` y las herramientas
  del template, y se abrió la misma interfaz de chat con `GradioUI`. Se reemplazó la herramienta vacía del
  template por `days_until`, creada por mí. En las pruebas el agente se recuperó solo de varios errores de
  formato, reformuló búsquedas que no daban resultados y combinó dos herramientas en un solo paso gracias
  a que es un Code Agent.
- **Modelos:** `gemini-flash-latest` devolvió error 503 por alta demanda, por lo que se usó `gemini-4-hydrogen`.
