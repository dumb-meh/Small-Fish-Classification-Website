# Fish Classification Website

A Flask-based web application for fish classification with an AI-powered chatbot that answers questions about small fishes in Bangladesh.

## Features

- 🐠 Serve multiple HTML pages (index, about, how-it-works, fish-database)
- 🤖 AI-powered chatbot integrated with backend
- 🔒 fish-classification-website.html is blocked from public access
- 💬 Real-time chat with conversation history
- 🌐 RESTful API for chatbot interactions

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- GROQ API Key (for chatbot functionality)

### Installation

1. Navigate to the Backend directory:
```powershell
cd Backend
```

2. Install required Python packages:
```powershell
pip install -r requirements.txt
```

3. Set your GROQ API key as an environment variable:
```powershell
$env:GROQ_API_KEY='your-groq-api-key-here'
```

### Running the Server

1. Make sure you're in the Backend directory:
```powershell
cd Backend
```

2. Run the Flask application:
```powershell
python main.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Available Pages

- **Home**: `http://localhost:5000/` (index.html)
- **About**: `http://localhost:5000/about.html`
- **How It Works**: `http://localhost:5000/how-it-works.html`
- **Fish Database**: `http://localhost:5000/fish-database.html`

Note: `fish-classification-website.html` is intentionally blocked and will return a 404 error.

## API Endpoints

### POST /api/chat
Send a message to the chatbot.

**Request Body:**
```json
{
  "message": "What are common small fishes in Bangladesh?",
  "session_id": "optional_session_id"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Bot response here...",
  "session_id": "session_id"
}
```

### POST /api/chat/clear
Clear chat history for a session.

**Request Body:**
```json
{
  "session_id": "optional_session_id"
}
```

### GET /api/chat/history
Get chat history for a session.

**Query Parameter:**
- `session_id` (optional, defaults to 'default')

**Response:**
```json
{
  "success": true,
  "history": [
    {
      "role": "user",
      "content": "message",
      "timestamp": "2026-01-02T..."
    }
  ],
  "session_id": "session_id"
}
```

### GET /health
Health check endpoint.

### POST /api/classify
Upload an image for classification. Accepts multipart/form-data with field `image` and optional `session_id`.

**Response:**
```json
{
  "success": true,
  "label": "puti",
  "confidence": 0.912,
  "fish": { /* static fish data from Backend/database/fish_data.py */ }
}
```

## Chatbot Features

- **Specialized Knowledge**: Focuses exclusively on small fishes in Bangladesh
- **Conversation History**: Maintains context across multiple messages
- **Session Management**: Supports multiple chat sessions
- **Persistent Storage**: Chat history is saved to JSON files

## Project Structure

```
Small Fish Website/
├── Backend/
│   ├── main.py              # Flask server
│   ├── backend.py           # Chatbot logic
│   ├── requirements.txt     # Python dependencies
│   └── chat_history_*.json  # Chat history files (auto-generated)
├── Frontend/
│   ├── index.html
│   ├── about.html
│   ├── how-it-works.html
│   ├── fish-database.html
│   ├── header.html
│   ├── footer.html
│   └── fish-classification-website.html (blocked)
└── .gitignore
```

## Troubleshooting

### Chatbot not responding
- Ensure GROQ_API_KEY environment variable is set
- Check if the backend server is running
- Look for error messages in the browser console

### Server won't start
- Check if port 5000 is already in use
- Verify all dependencies are installed
- Ensure you're running the command from the Backend directory

### 404 errors for HTML pages
- Verify the HTML files exist in the Frontend directory
- Check file names match exactly (case-sensitive on some systems)

## Development

The server runs in debug mode by default, which means:
- Auto-reload on file changes
- Detailed error messages
- Access from any network interface (0.0.0.0)

For production deployment, disable debug mode and use a production WSGI server like Gunicorn.

## License

This project is for educational purposes.
