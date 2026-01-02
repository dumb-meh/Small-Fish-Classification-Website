import os
import json
from typing import List, Dict
from groq import Groq
from datetime import datetime

class CachedChatHistory:
    def __init__(self, session_id: str = "default"):
        api_key = os.getenv("GROQ_API_KEY")
        print(f"[Backend] Initializing CachedChatHistory for session: {session_id}")
        print(f"[Backend] API Key present: {bool(api_key)}")
        print(f"[Backend] API Key length: {len(api_key) if api_key else 0}")
        
        try:
            self.client = Groq(api_key=api_key)
            print(f"[Backend] Groq client initialized successfully")
        except Exception as e:
            print(f"[Backend] ERROR initializing Groq client: {e}")
            raise
            
        self.model = "llama-3.1-8b-instant"
        self.session_id = session_id
        self.cache_file = f"chat_history_{session_id}.json"
        
        # STRICT SYSTEM PROMPT
        self.system_prompt = """You are an expert on small fishes in Bangladesh. 
Your ONLY purpose is to answer questions about small fishes found in Bangladesh.
You MUST follow these rules STRICTLY:
1. ONLY answer questions about small fishes in Bangladesh
2. For any question NOT about small fishes in Bangladesh, respond: "Sorry, I can only answer questions about small fishes in Bangladesh. Please ask about small fishes."
3. Do not engage in any other conversation topics
4. Do not answer questions about other countries' fishes unless specifically compared to Bangladesh
5. Stay strictly on topic at all times

Examples of acceptable questions:
- "What are the common small fishes in Bangladesh?"
- "Tell me about Puti fish"
- "How are small fishes farmed in Bangladesh?"

Examples of unacceptable questions (you must politely refuse):
- "What's the weather like today?" → "Sorry, I can only answer questions about small fishes in Bangladesh."
- "Tell me a joke" → "Sorry, I can only answer questions about small fishes in Bangladesh."
- "What about sharks?" → "Sorry, sharks are not small fishes. I can only answer about small fishes in Bangladesh."
"""
        
        self.conversation_history: List[Dict] = self._load_history()
        
    def _load_history(self) -> List[Dict]:
        """Load chat history from file"""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                return data.get("messages", [])
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize with system prompt
            return [{"role": "system", "content": self.system_prompt}]
    
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
        # Keep last 15 messages + system prompt to avoid context overflow
        if len(self.conversation_history) > 16:  # 1 system + 15 conversation
            # Keep system prompt and recent messages
            system_msg = self.conversation_history[0]
            recent_msgs = self.conversation_history[-15:]
            self.conversation_history = [system_msg] + recent_msgs
        self._save_history()
    
    def get_response(self, user_message: str) -> str:
        """Get response using full chat history"""
        print(f"[Backend] get_response called with message: {user_message[:50]}...")
        
        # Add user message to history
        self.add_to_history("user", user_message)
        print(f"[Backend] User message added to history. Total messages: {len(self.conversation_history)}")
        
        try:
            print(f"[Backend] Calling Groq API with model: {self.model}")
            
            # Prepare messages for API - system prompt + last 10 conversation messages
            system_msg = [msg for msg in self.conversation_history if msg["role"] == "system"]
            conversation_msgs = [msg for msg in self.conversation_history if msg["role"] != "system"]
            
            # Take only last 10 conversation messages (5 exchanges)
            recent_conversation = conversation_msgs[-10:] if len(conversation_msgs) > 10 else conversation_msgs
            
            # Combine system prompt with recent conversation
            messages_to_send = system_msg + recent_conversation
            
            # Remove timestamp field for API
            api_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages_to_send
            ]
            
            print(f"[Backend] Total messages in full history: {len(self.conversation_history)}")
            print(f"[Backend] Sending to API: 1 system + {len(recent_conversation)} conversation messages = {len(api_messages)} total")
            
            # Call API with system prompt + recent conversation
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=512,
                top_p=0.9
            )
            
            print(f"[Backend] API call successful")
            assistant_response = completion.choices[0].message.content
            print(f"[Backend] Assistant response received: {assistant_response[:50]}...")
            
            # Add assistant response to history
            self.add_to_history("assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[Backend] ERROR in get_response: {error_msg}")
            print(f"[Backend] Exception type: {type(e).__name__}")
            print(f"[Backend] Full error: {repr(e)}")
            return error_msg
    
    def clear_history(self):
        """Clear conversation history (keep system prompt)"""
        system_prompt = [msg for msg in self.conversation_history if msg["role"] == "system"]
        if not system_prompt:
            system_prompt = [{"role": "system", "content": self.system_prompt}]
        self.conversation_history = system_prompt
        self._save_history()
        print(f"Chat history cleared for session: {self.session_id} (kept system prompt)")
    
    def show_history(self):
        """Display conversation history"""
        print(f"\n=== Chat History ({self.session_id}) ===")
        for i, msg in enumerate(self.conversation_history):
            role = msg["role"].upper()
            if role == "SYSTEM":
                content = "System prompt loaded..."
            else:
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"{i}. {role}: {content}")
        print("=" * 40)
    
    def show_system_prompt(self):
        """Display the system prompt"""
        print("\n=== SYSTEM PROMPT ===")
        print(self.system_prompt)
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
    session = manager.get_session("fish_expert")
    
    print("=" * 60)
    print("BANGLADESH SMALL FISH EXPERT CHATBOT")
    print("I can ONLY answer questions about small fishes in Bangladesh.")
    print("For anything else, I will politely refuse.")
    print("=" * 60)
    print("Commands: 'quit', 'clear', 'history', 'system', 'switch [session]'")
    print("-" * 60)
    
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
            elif user_input.lower() == 'system':
                session.show_system_prompt()
                continue
            elif user_input.startswith('switch '):
                new_session = user_input.split(' ', 1)[1]
                session = manager.get_session(new_session)
                print(f"Switched to session: {new_session}")
                print("I can ONLY answer questions about small fishes in Bangladesh.")
                continue
            
            response = session.get_response(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break