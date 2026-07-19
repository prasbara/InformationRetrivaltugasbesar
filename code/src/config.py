"""
Module: config.py
Purpose: Manages loading and saving application configurations from/to config.yaml, and retrieving API keys from environment variables.
Inputs: Settings dictionary or key updates.
Outputs: Dictionary of current configuration settings.
Workflow: Defines default settings, loads configurations from config.yaml using a fallback merge with defaults, saves updated configurations back to YAML, and retrieves API keys.
Dependencies: os, yaml, typing, dotenv.
Complexity: Time: O(1) file reads/writes; Space: O(1) configuration data cache.
"""
import os
import yaml
from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the nearest .env file in parent directories
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

DEFAULT_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_provider": "ollama",
    "llm_model": "llama3",
    "openrouter_model": "meta-llama/llama-3-8b-instruct:free",
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 1024,
    "repeat_penalty": 1.1,
    "context_window": 4096,
    "chunk_size": 700,
    "chunk_overlap": 150,
    "top_k_retrieval": 5,
    "database_path": "vector_db/chroma/"
}

def load_config() -> Dict[str, Any]:
    """Loads config from config.yaml, falling back to defaults if file is missing."""
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            # Merge with defaults to ensure all keys exist
            merged_config = DEFAULT_CONFIG.copy()
            merged_config.update(config)
            return merged_config
    except Exception as e:
        print(f"Error loading config.yaml: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()

def save_config(config_data: Dict[str, Any]) -> bool:
    """Saves updated configurations back to config.yaml."""
    try:
        # Resolve absolute path for database_path in the config if needed
        # but save it relative in yaml
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error saving config.yaml: {e}")
        return False

def get_openrouter_api_key() -> str:
    """Retrieve OpenRouter API key from environment variables."""
    return os.getenv("OPENROUTER_API_KEY", "")
