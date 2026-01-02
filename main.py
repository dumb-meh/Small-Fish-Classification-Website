from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from Backend.backend import ChatSessionManager

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, 
            template_folder='Frontend',
            static_folder='Frontend')
CORS(app)  # Enable CORS for all routes

# Initialize chat session manager
chat_manager = ChatSessionManager()

@app.route('/')
def index():
    """Serve the main index.html page"""
    return send_from_directory('Frontend', 'index.html')

@app.route('/<path:filename>')
def serve_html(filename):
    """Serve HTML files from Frontend folder (except fish-classification-website.html)"""
    # Block access to fish-classification-website.html
    if filename == 'fish-classification-website.html':
        return "File not found", 404
    
    # Serve other HTML and static files
    if os.path.exists(os.path.join('Frontend', filename)):
        return send_from_directory('Frontend', filename)
    return "File not found", 404

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat requests from the frontend chatbot
    Expected JSON format: {"message": "user message", "session_id": "optional_session_id"}
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message']
        session_id = data.get('session_id', 'default')
        
        # Get or create chat session
        session = chat_manager.get_session(session_id)
        
        # Get response from the chatbot
        bot_response = session.get_response(user_message)
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """
    Clear chat history for a specific session
    Expected JSON format: {"session_id": "session_id"}
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        session = chat_manager.get_session(session_id)
        session.clear_history()
        
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_history():
    """
    Get chat history for a specific session
    Query parameter: session_id (optional, defaults to 'default')
    """
    try:
        session_id = request.args.get('session_id', 'default')
        session = chat_manager.get_session(session_id)
        
        # Get conversation history (excluding system prompt)
        history = [msg for msg in session.conversation_history if msg['role'] != 'system']
        
        return jsonify({
            'success': True,
            'history': history,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Fish Classification Website'
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY environment variable is not set!")
        print("The chatbot will not work without this API key.")
        print("Set it using: $env:GROQ_API_KEY='your-key-here' (PowerShell)")
    
    print("=" * 60)
    print("Fish Classification Website Server")
    print("=" * 60)
    print("Server running at: http://localhost:5000")
    print("Available pages:")
    print("  - http://localhost:5000/ (index.html)")
    print("  - http://localhost:5000/about.html")
    print("  - http://localhost:5000/how-it-works.html")
    print("  - http://localhost:5000/fish-database.html")
    print("API endpoints:")
    print("  - POST /api/chat (send chatbot messages)")
    print("  - POST /api/chat/clear (clear chat history)")
    print("  - GET /api/chat/history (get chat history)")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
