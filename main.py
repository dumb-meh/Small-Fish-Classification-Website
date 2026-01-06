from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from Backend.backend import ChatSessionManager
from Backend.image_classification import classify_image, model_status
from Backend.database.fish_data import get_fish_data

# Load environment variables from .env file
load_dotenv()

print("\n" + "="*60)
print("[Flask] Starting Fish Classification Website")
print("="*60)
api_key = os.getenv("GROQ_API_KEY")
print(f"[Flask] GROQ_API_KEY loaded: {bool(api_key)}")
if api_key:
    print(f"[Flask] API Key length: {len(api_key)} characters")
    print(f"[Flask] API Key starts with: {api_key[:10]}...")
else:
    print("[Flask] WARNING: GROQ_API_KEY not found!")
print("="*60 + "\n")

app = Flask(__name__, 
            template_folder='Frontend',
            static_folder='Frontend')
CORS(app)  # Enable CORS for all routes

# Initialize chat session manager
print("[Flask] Initializing ChatSessionManager...")
try:
    chat_manager = ChatSessionManager()
    print("[Flask] ChatSessionManager initialized successfully\n")
except Exception as e:
    print(f"[Flask] ERROR initializing ChatSessionManager: {e}\n")
    raise

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
    print("\n" + "="*60)
    print("[Flask] /api/chat endpoint called")
    print("="*60)
    
    try:
        data = request.get_json()
        print(f"[Flask] Request data received: {data}")
        
        if not data or 'message' not in data:
            print("[Flask] ERROR: No message in request data")
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message']
        session_id = data.get('session_id', 'default')
        
        print(f"[Flask] User message: {user_message}")
        print(f"[Flask] Session ID: {session_id}")
        
        # Get or create chat session
        print(f"[Flask] Getting chat session...")
        session = chat_manager.get_session(session_id)
        print(f"[Flask] Chat session obtained: {session}")
        
        # Get response from the chatbot
        print(f"[Flask] Calling get_response...")
        bot_response = session.get_response(user_message)
        print(f"[Flask] Bot response received: {bot_response[:100]}...")
        
        response_data = {
            'success': True,
            'response': bot_response,
            'session_id': session_id
        }
        print(f"[Flask] Sending successful response")
        print("="*60 + "\n")
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"[Flask] ERROR in /api/chat: {str(e)}")
        print(f"[Flask] Exception type: {type(e).__name__}")
        print(f"[Flask] Full error: {repr(e)}")
        import traceback
        print(f"[Flask] Traceback:")
        traceback.print_exc()
        print("="*60 + "\n")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classify', methods=['POST'])
def classify():
    """Handle image classification requests.
    Expects multipart/form-data with field 'image' and optional 'session_id'.
    """
    print("\n" + "="*60)
    print("[Flask] /api/classify endpoint called")
    print("[Flask] Request method:", request.method)
    print("[Flask] Request content type:", request.content_type)
    print("[Flask] Request files:", list(request.files.keys()))
    print("[Flask] Request form:", dict(request.form))
    print("="*60)

    try:
        if 'image' not in request.files:
            print("[Flask] ERROR: No image file in request")
            print("[Flask] Available files:", list(request.files.keys()))
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        img = request.files['image']
        print(f"[Flask] Image file received: {img.filename}")
        print(f"[Flask] Image content type: {img.content_type}")
        print(f"[Flask] Image file object: {img}")
        
        session_id = request.form.get('session_id', 'default')
        print(f"[Flask] Session ID: {session_id}")

        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            print(f"[Flask] Created uploads directory: {uploads_dir}")
        else:
            print(f"[Flask] Using existing uploads directory: {uploads_dir}")

        # Save uploaded file
        filename = f"{session_id}_{int(__import__('time').time())}_{img.filename}"
        file_path = os.path.join(uploads_dir, filename)
        print(f"[Flask] Saving file to: {file_path}")
        img.save(file_path)
        print(f"[Flask] File saved successfully")
        print(f"[Flask] File exists: {os.path.exists(file_path)}")
        print(f"[Flask] File size: {os.path.getsize(file_path)} bytes")

        # Classify
        print(f"[Flask] Starting classification...")
        label, confidence, method = classify_image(file_path)
        print(f"[Flask] Classification complete!")
        print(f"[Flask] Result - Label: {label}, Confidence: {confidence}, Method: {method}")

        # Get static fish data if available
        print(f"[Flask] Fetching fish data for label: {label}")
        fish_info = get_fish_data(label)
        print(f"[Flask] Fish data retrieved: {bool(fish_info)}")
        if fish_info:
            print(f"[Flask] Fish name: {fish_info.get('name_en', 'N/A')}")

        response = {
            'success': True,
            'label': label,
            'confidence': confidence,
            'method': method,
            'fish': fish_info
        }
        print(f"[Flask] Sending response with success=True")
        print(f"[Flask] Response keys: {list(response.keys())}")
        print(f"[Flask] /api/classify completed successfully")
        print("="*60 + "\n")
        
        # Also print model status for debug
        status = model_status()
        print(f"[Flask] Model status: {status}")
        return jsonify(response)

    except Exception as e:
        print(f"[Flask] ERROR in /api/classify: {e}")
        print(f"[Flask] Exception type: {type(e).__name__}")
        import traceback
        print(f"[Flask] Full traceback:")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-status', methods=['GET'])
def model_status_endpoint():
    """Return model loading diagnostics"""
    try:
        status = model_status()
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        print(f"[Flask] ERROR in /api/model-status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
