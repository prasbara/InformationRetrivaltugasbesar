"""
Module: rag.py
Purpose: Integrates local (Ollama) and cloud (OpenRouter) LLM providers to stream context-aware responses.
Inputs: LLM provider (string), model name (string), prompt (string), hyperparameters (dict), API key (string).
Outputs: Generator yielding streamed text response chunks.
Workflow: Checks service status, structures request payloads, sends API queries via requests, and parses stream lines to yield content chunks.
Dependencies: json, requests, typing, src.logger.
Complexity: Time: O(T) where T is generation time; Space: O(1) stream buffer.
"""
import json
import requests
from typing import Generator, Dict, Any, List

from src.logger import log_system

OLLAMA_BASE_URL = "http://localhost:11434"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def check_ollama_status() -> bool:
    """Checks if the local Ollama service is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_installed_ollama_models() -> List[str]:
    """Retrieves a list of model names currently installed on the local Ollama instance."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            return [model["name"] for model in models_data]
        return []
    except Exception as e:
        log_system(f"Failed to fetch Ollama models: {str(e)}", "warning")
        return []

def query_llm_stream(
    provider: str,
    model: str,
    prompt: str,
    params: Dict[str, Any],
    api_key: str = ""
) -> Generator[str, None, None]:
    """Queries either Ollama or OpenRouter and streams the text response."""
    
    # Standardize options/parameters
    temperature = params.get("temperature", 0.1)
    top_p = params.get("top_p", 0.9)
    top_k = params.get("top_k", 40)
    max_tokens = params.get("max_tokens", 1024)
    repeat_penalty = params.get("repeat_penalty", 1.1)
    context_window = params.get("context_window", 4096)

    if provider.lower() == "ollama":
        if not check_ollama_status():
            raise ConnectionError("Ollama service is not running. Please make sure Ollama is started locally.")
            
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens,
                "repeat_penalty": repeat_penalty,
                "num_ctx": context_window
            }
        }
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=10)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")
        except requests.exceptions.RequestException as e:
            log_system(f"Ollama request error: {str(e)}", "error")
            raise ConnectionError(f"Ollama connection error: {str(e)}")

    elif provider.lower() == "openrouter":
        if not api_key:
            raise ValueError("OpenRouter API key is missing. Please provide the key in the settings or environment.")
            
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Local AI Campus Assistant"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True, timeout=15)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:].strip()
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_content)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                yield delta.get("content", "")
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            log_system(f"OpenRouter request error: {str(e)}", "error")
            raise ConnectionError(f"OpenRouter connection error: {str(e)}")
            
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def query_llm_non_stream(
    provider: str,
    model: str,
    prompt: str,
    params: Dict[str, Any],
    api_key: str = ""
) -> str:
    """Queries either Ollama or OpenRouter and returns the full text response."""
    response_text = ""
    for chunk in query_llm_stream(provider, model, prompt, params, api_key):
        response_text += chunk
    return response_text
