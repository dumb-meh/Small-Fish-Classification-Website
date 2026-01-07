// Shared Chatbot JavaScript for all pages
// This file handles the chatbot UI and API communication

let chatOpen = false;
let messageCount = 0;
let sessionId = null;

// Generate unique session ID
function generateSessionId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 15);
    const uuid = `${timestamp}-${random}`;
    console.log('[Chatbot] Generated session ID:', uuid);
    return uuid;
}

// Get or create session ID
function getSessionId() {
    if (!sessionId) {
        // Try to get from sessionStorage first
        sessionId = sessionStorage.getItem('chatbot_session_id');
        if (!sessionId) {
            sessionId = generateSessionId();
            sessionStorage.setItem('chatbot_session_id', sessionId);
            console.log('[Chatbot] Created new session ID:', sessionId);
        } else {
            console.log('[Chatbot] Retrieved existing session ID:', sessionId);
        }
    }
    return sessionId;
}

// Load chatbot HTML structure
function loadChatbotComponents() {
    console.log('[Chatbot] Loading chatbot components...');
    
    const chatbotHTML = `
        <!-- Chatbot Popup -->
        <div id="chatbotPopup" class="fixed bottom-24 right-6 w-96 bg-white rounded-2xl shadow-2xl z-50 hidden transform transition-all duration-300">
            <!-- Chatbot Header -->
            <div class="bg-gradient-to-r from-primary to-secondary text-white p-4 rounded-t-2xl flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                        <i class="fas fa-robot text-xl"></i>
                    </div>
                    <div>
                        <h3 class="font-bold text-lg">FishAI Assistant</h3>
                        <p class="text-blue-100 text-sm">Online - Ask me about fish!</p>
                    </div>
                </div>
                <button id="closeChatbot" class="text-white hover:text-blue-200">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>
            
            <!-- Chat Messages -->
            <div id="chatMessages" class="h-96 overflow-y-auto p-4 space-y-4">
                <!-- Welcome Message -->
                <div class="chat-message bg-blue-50 rounded-2xl rounded-tl-none p-4">
                    <div class="flex items-start space-x-2">
                        <div class="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-robot text-white text-sm"></i>
                        </div>
                        <div>
                            <p class="font-medium text-dark mb-1">FishAI Assistant</p>
                            <p class="text-gray-700">Hello! I'm your Bangladesh small fish expert assistant. I can help you with:</p>
                            <ul class="list-disc pl-4 text-gray-700 mt-1 space-y-1">
                                <li>Information about small fishes found in Bangladesh</li>
                                <li>Fish habitats and behavior in Bangladesh</li>
                                <li>Local fish species identification</li>
                                <li>Fish farming practices in Bangladesh</li>
                            </ul>
                            <p class="text-gray-700 mt-2">What would you like to know about Bangladesh's small fishes?</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Chat Input -->
            <div class="border-t p-4">
                <div class="flex space-x-2">
                    <div class="flex-1 relative">
                        <input type="text" id="chatInput" placeholder="Type your question about fish..." class="w-full border rounded-xl py-3 px-4 pr-12 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <button id="sendMessage" class="bg-primary text-white px-6 rounded-xl hover:bg-blue-700 transition-colors">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
                <p class="text-xs text-gray-400 mt-2">Powered by AI - Ask about Bangladesh small fishes</p>
            </div>
        </div>

        <!-- Chatbot Toggle Button -->
        <button id="chatbotToggleFixed" class="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-primary to-secondary text-white rounded-full shadow-lg hover:shadow-xl transition-shadow flex items-center justify-center chatbot-toggle">
            <i class="fas fa-robot text-xl"></i>
            <span class="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white text-xs rounded-full flex items-center justify-center animate-pulse hidden" id="chatNotification">1</span>
        </button>
    `;
    
    const container = document.getElementById('chatbot-components');
    if (container) {
        container.innerHTML = chatbotHTML;
        console.log('[Chatbot] Chatbot HTML loaded successfully');
        initChatbot();
    } else {
        console.error('[Chatbot] ERROR: chatbot-components container not found!');
    }
}

// Initialize chatbot functionality
function initChatbot() {
    console.log('[Chatbot] Initializing chatbot functionality...');
    
    const chatbotPopup = document.getElementById('chatbotPopup');
    const chatbotToggle = document.getElementById('chatbotToggle');
    const chatbotToggleFixed = document.getElementById('chatbotToggleFixed');
    const closeChatbot = document.getElementById('closeChatbot');
    const chatInput = document.getElementById('chatInput');
    const sendMessage = document.getElementById('sendMessage');

    // Toggle chatbot from header button
    if (chatbotToggle) {
        console.log('[Chatbot] Header toggle button found, attaching listener');
        chatbotToggle.addEventListener('click', toggleChatbot);
    } else {
        console.warn('[Chatbot] Header toggle button not found');
    }

    // Toggle chatbot from fixed button
    if (chatbotToggleFixed) {
        console.log('[Chatbot] Fixed toggle button found, attaching listener');
        chatbotToggleFixed.addEventListener('click', toggleChatbot);
    }

    // Close chatbot
    if (closeChatbot) {
        closeChatbot.addEventListener('click', () => {
            console.log('[Chatbot] Close button clicked');
            chatbotPopup.classList.add('hidden');
            chatOpen = false;
        });
    }

    // Send message button
    if (sendMessage) {
        console.log('[Chatbot] Send button found, attaching listener');
        sendMessage.addEventListener('click', sendChatMessage);
    }

    // Enter key to send message
    if (chatInput) {
        console.log('[Chatbot] Chat input found, attaching listener');
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('[Chatbot] Enter key pressed');
                sendChatMessage();
            }
        });
    }

    console.log('[Chatbot] Initialization complete!');
}

// Toggle chatbot visibility
function toggleChatbot() {
    console.log('[Chatbot] Toggle chatbot called, current state:', chatOpen ? 'open' : 'closed');
    const chatbotPopup = document.getElementById('chatbotPopup');
    const chatInput = document.getElementById('chatInput');
    
    chatOpen = !chatOpen;
    
    if (chatOpen) {
        console.log('[Chatbot] Opening chatbot...');
        chatbotPopup.classList.remove('hidden');
        chatbotPopup.classList.add('animate-float');
        if (chatInput) chatInput.focus();
    } else {
        console.log('[Chatbot] Closing chatbot...');
        chatbotPopup.classList.add('hidden');
        chatbotPopup.classList.remove('animate-float');
    }
}

// Send chat message to backend API
async function sendChatMessage() {
    console.log('[Chatbot] sendChatMessage called');
    
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const message = chatInput.value.trim();
    
    if (!message) {
        console.warn('[Chatbot] Empty message, ignoring');
        return;
    }

    console.log('[Chatbot] User message:', message);

    // Add user message
    addChatMessage(message, 'user');
    chatInput.value = '';

    // Add typing indicator
    const typingIndicator = addTypingIndicator();

    try {
        console.log('[Chatbot] Sending request to /api/chat...');
        const requestBody = {
            message: message,
            session_id: getSessionId()
        };
        console.log('[Chatbot] Request body:', requestBody);
        
        // Send message to backend API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        console.log('[Chatbot] Response status:', response.status);
        console.log('[Chatbot] Response ok:', response.ok);

        const data = await response.json();
        console.log('[Chatbot] Response data:', data);

        // Remove typing indicator
        typingIndicator.remove();

        if (data.success) {
            console.log('[Chatbot] Success! AI response:', data.response);
            // Add AI response
            addChatMessage(data.response, 'ai');
        } else {
            console.error('[Chatbot] API returned error:', data.error);
            addChatMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'), 'ai');
        }
    } catch (error) {
        console.error('[Chatbot] Fetch error:', error);
        console.error('[Chatbot] Error details:', error.message);
        console.error('[Chatbot] Error stack:', error.stack);
        
        typingIndicator.remove();
        addChatMessage('Sorry, I could not connect to the server. Error: ' + error.message, 'ai');
    }
}

// Add message to chat
function addChatMessage(text, sender) {
    console.log('[Chatbot] Adding message:', sender, '-', text.substring(0, 50) + '...');
    
    const chatMessages = document.getElementById('chatMessages');
    messageCount++;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender === 'user' ? 'ml-auto bg-primary text-white rounded-2xl rounded-tr-none' : 'bg-blue-50 rounded-2xl rounded-tl-none'}`;
    
    messageDiv.innerHTML = `
        <div class="flex items-start space-x-2 ${sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''}">
            <div class="w-8 h-8 ${sender === 'user' ? 'bg-white/20' : 'bg-primary'} rounded-full flex items-center justify-center flex-shrink-0">
                <i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'} text-${sender === 'user' ? 'white' : 'white'} text-sm"></i>
            </div>
            <div class="${sender === 'user' ? 'text-right' : ''}">
                <p class="font-medium ${sender === 'user' ? 'text-blue-100' : 'text-dark'} mb-1">${sender === 'user' ? 'You' : 'FishAI Assistant'}</p>
                <p class="${sender === 'user' ? 'text-blue-100' : 'text-gray-700'}">${text}</p>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add typing indicator
function addTypingIndicator() {
    console.log('[Chatbot] Adding typing indicator...');
    
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message bg-blue-50 rounded-2xl rounded-tl-none';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="flex items-start space-x-2">
            <div class="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                <i class="fas fa-robot text-white text-sm"></i>
            </div>
            <div>
                <p class="font-medium text-dark mb-1">FishAI Assistant</p>
                <div class="flex space-x-1">
                    <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0s"></span>
                    <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></span>
                    <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></span>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

// Handle image uploads in chat: send to server for classification
function handleChatImageUpload(file) {
    console.log('[Chatbot] handleChatImageUpload called');

    if (!file.type.match('image.*')) {
        addChatMessage('Please upload an image file (JPG, PNG, GIF, etc.)', 'ai');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const chatMessages = document.getElementById('chatMessages');
        // Add user message with image
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message ml-auto bg-primary text-white rounded-2xl rounded-tr-none p-3';
        messageDiv.innerHTML = `
            <div class="flex items-start space-x-2 flex-row-reverse space-x-reverse">
                <div class="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-user text-white text-sm"></i>
                </div>
                <div class="text-right">
                    <p class="font-medium text-blue-100 mb-1">You</p>
                    <img src="${e.target.result}" alt="Uploaded fish image" class="max-w-xs rounded-lg mb-2">
                    <p class="text-blue-100">Can you identify this fish?</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Now send image to backend
        sendImageForClassification(file);
    };
    reader.readAsDataURL(file);
}

// Expose handleChatImageUpload globally with a unique name to avoid conflicts
if (typeof window !== 'undefined') {
    window.chatbotHandleImageUpload = handleChatImageUpload;
    console.log('[Chatbot] Exposed chatbotHandleImageUpload globally');
}

async function sendImageForClassification(file) {
    console.log('[Chatbot] Sending image to /api/classify');
    const typingIndicator = addTypingIndicator();

    try {
        const form = new FormData();
        form.append('image', file);
        form.append('session_id', getSessionId());

        const resp = await fetch('/api/classify', {
            method: 'POST',
            body: form
        });

        console.log('[Chatbot] /api/classify status:', resp.status);
        const data = await resp.json();
        console.log('[Chatbot] /api/classify response:', data);

        typingIndicator.remove();

        if (data.success) {
            const label = data.label || 'unknown';
            const conf = data.confidence || 0;
            const method = data.method || 'unknown';

            // Build a message with classification results to send to the LLM
            const classificationMessage = `I uploaded an image of a fish. The AI model identified it as "${label}" with ${(conf*100).toFixed(1)}% confidence${method === 'fallback' ? ' (using fallback classifier)' : ''}. Can you tell me more about this fish?`;
            
            console.log('[Chatbot] Sending classification result to LLM:', classificationMessage);
            
            // Send classification result to chatbot API
            const chatResp = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: classificationMessage,
                    session_id: getSessionId()
                })
            });

            const chatData = await chatResp.json();
            console.log('[Chatbot] LLM response:', chatData);

            if (chatData.success) {
                addChatMessage(chatData.response, 'ai');
            } else {
                addChatMessage('Sorry, I encountered an error getting information about this fish.', 'ai');
            }
        } else {
            addChatMessage('Sorry, I could not classify the image: ' + (data.error || 'Unknown error'), 'ai');
        }
    } catch (err) {
        console.error('[Chatbot] Error sending image:', err);
        typingIndicator.remove();
        addChatMessage('Sorry, there was a problem uploading the image: ' + err.message, 'ai');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('[Chatbot] DOMContentLoaded event fired');
    
    // Wait a bit for header to load
    setTimeout(() => {
        console.log('[Chatbot] Attempting to load chatbot components...');
        loadChatbotComponents();
    }, 500);

    // Fetch model status for debugging
    fetch('/api/model-status')
        .then(r => r.json())
        .then(data => {
            console.log('[Chatbot][ModelStatus] /api/model-status:', data);
        })
        .catch(err => {
            console.warn('[Chatbot][ModelStatus] Unable to fetch model status:', err);
        });
});
