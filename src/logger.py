import os
import logging
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Sets up a logger with a rotating file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if logger is already set up
    if not logger.handlers:
        log_path = os.path.join(LOGS_DIR, log_file)
        
        # Max size 5MB, keeping 3 backups
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        # Prevent logs from propagating to the root logger
        logger.propagate = False
        
    return logger

# Create specific loggers
system_logger = setup_logger("system", "system.log")
chat_logger = setup_logger("chat", "chat.log")
retrieval_logger = setup_logger("retrieval", "retrieval.log")

def log_system(message: str, level: str = "info"):
    """Logs system level messages (errors, status updates)."""
    if level.lower() == "error":
        system_logger.error(message)
    elif level.lower() == "warning":
        system_logger.warning(message)
    else:
        system_logger.info(message)

def log_chat(user_msg: str, response_msg: str, provider: str, model: str):
    """Logs details about a chat transaction."""
    chat_logger.info(f"Provider: {provider} | Model: {model}")
    chat_logger.info(f"User: {user_msg}")
    chat_logger.info(f"Assistant: {response_msg}")
    chat_logger.info("-" * 50)

def log_retrieval(query: str, top_k: int, results: list):
    """Logs details about document retrievals and their similarity scores."""
    retrieval_logger.info(f"Query: {query} | Top_K: {top_k}")
    for idx, doc in enumerate(results):
        meta = doc.metadata
        # Support both Document objects and manual lists
        doc_name = meta.get('source', 'Unknown')
        page = meta.get('page', 0)
        score = meta.get('score', 'N/A')
        retrieval_logger.info(f"  Match {idx+1}: Doc={doc_name}, Page={page}, Score={score}")
    retrieval_logger.info("=" * 50)
