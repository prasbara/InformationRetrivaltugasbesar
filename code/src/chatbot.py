"""
Module: chatbot.py
Purpose: Manages chat session history using LangChain's ConversationBufferMemory, keeping track of conversation history and logging chat logs.
Inputs: User message (string), AI message (string), LLM provider (string), LLM model (string).
Outputs: List of LangChain messages, formatted history string.
Workflow: Initializes conversation memory, adds user and AI messages dynamically, retrieves message history, and formats/cleans history strings.
Dependencies: langchain_classic.memory, src.logger.
Complexity: Time: O(1) for adding/retrieving, O(N) for string formatting; Space: O(N) where N is the number of chat turns.
"""
from langchain_classic.memory import ConversationBufferMemory
from src.logger import log_chat

class ChatbotManager:
    """Manages chat session history using LangChain's ConversationBufferMemory."""
    
    def __init__(self):
        # We set return_messages=True to get list of message objects, 
        # and memory_key="chat_history" to match LangChain standards.
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

    def add_user_message(self, message: str) -> None:
        """Saves a user message to the memory."""
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        """Saves an AI response to the memory."""
        self.memory.chat_memory.add_ai_message(message)

    def get_history(self) -> list:
        """Retrieves a list of messages (HumanMessage and AIMessage objects)."""
        return self.memory.chat_memory.messages

    def get_history_string(self) -> str:
        """Returns the history formatted as a single string."""
        return self.memory.load_memory_variables({}).get("chat_history", "")

    def clear_history(self) -> None:
        """Clears the conversational memory."""
        self.memory.clear()

    def process_chat_turn(
        self, 
        user_message: str, 
        response_text: str, 
        provider: str, 
        model: str
    ) -> None:
        """Saves messages to memory and logs the conversation."""
        self.add_user_message(user_message)
        self.add_ai_message(response_text)
        log_chat(user_message, response_text, provider, model)
