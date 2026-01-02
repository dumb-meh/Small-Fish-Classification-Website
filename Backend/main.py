import os
import json
from typing import List, Dict
from groq import Groq
from datetime import datetime

class CachedChatHistory:
    def __init__(self, session_id: str = "default"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.session_id = session_id
        self.cache_file = f"chat_history_{session_id}.json"
        self.conversation_history: List[Dict] = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        """Load chat history from file"""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                return data.get("messages", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_history(self):
        """Save chat history to file"""
        history_data = {
            "session_id": self.session_id,
            "last_updated": datetime.now().isoformat(),
            "messages": self.conversation_history
        }
        with open(self.cache_file, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 20 messages to avoid context overflow
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        self._save_history()
    
    def get_response(self, user_message: str) -> str:
        """Get response using full chat history"""
        # Add user message to history
        self.add_to_history("user", user_message)
        
        try:
            # Call API with entire conversation history
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=1024
            )
            
            assistant_response = completion.choices[0].message.content
            
            # Add assistant response to history
            self.add_to_history("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self._save_history()
        print(f"Chat history cleared for session: {self.session_id}")
    
    def show_history(self):
        """Display conversation history"""
        print(f"\n=== Chat History ({self.session_id}) ===")
        for msg in self.conversation_history:
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"{role}: {content}")
        print("=" * 40)

# Multi-session manager
class ChatSessionManager:
    def __init__(self):
        self.sessions: Dict[str, CachedChatHistory] = {}
    
    def get_session(self, session_id: str) -> CachedChatHistory:
        """Get or create a chat session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = CachedChatHistory(session_id)
        return self.sessions[session_id]

# Usage example
if __name__ == "__main__":
    # Set your API key: export GROQ_API_KEY='your-key-here'
    manager = ChatSessionManager()
    
    # Start with default session or specify one
    session = manager.get_session("my_chat")
    
    print("Chatbot with history cache. Type 'quit' to exit.")
    print("Commands: 'clear', 'history', 'switch [session_name]'")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                session.clear_history()
                continue
            elif user_input.lower() == 'history':
                session.show_history()
                continue
            elif user_input.startswith('switch '):
                new_session = user_input.split(' ', 1)[1]
                session = manager.get_session(new_session)
                print(f"Switched to session: {new_session}")
                continue
            
            response = session.get_response(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break