import ollama

from app import config


def chat(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options={"temperature": temperature},
    )
    return response["message"]["content"]