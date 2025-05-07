/**
 * Chat Interface Script
 * ---------------------
 * Handles DOM interactions for a chat interface including:
 * - Sending user messages
 * - Receiving and formatting bot responses
 * - Displaying a typing indicator while waiting for the backend
 * - Toggle UI component handling
 * - Adds message timestamps and basic formatting for bot replies
 * 
 * Dependencies: Assumes backend API at /query endpoint (POST)
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const togglePill = document.querySelector('.toggle-pill');

    if (togglePill) {
        const toggleCircle = togglePill.querySelector('.toggle-circle');
        let isToggled = false;

        togglePill.addEventListener('click', () => {
            isToggled = !isToggled;
            if (isToggled) {
                toggleCircle.style.left = 'calc(100% - 30px)';
                toggleCircle.style.backgroundColor = '#000';
            } else {
                toggleCircle.style.left = '5px';
                toggleCircle.style.backgroundColor = '#000';
            }
        });
    }

    if (sendBtn && userInput) {
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        addMessage(message, 'user');
   
        userInput.value = '';
        
        fetchResponseFromBackend(message);
    }


    function addMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        
        if (sender === 'bot') {
            let formattedMessage = message
                
                .replace(/(\d+\.\s+)([^\d\n.]+)/g, '<strong>$1</strong>$2')
                .replace(/(Step \d+:)/gi, '<strong>$1</strong>')
                .replace(/\n/g, '<br>');
            
            
            formattedMessage = formattedMessage.replace(/<br><strong>(\d+\.)/g, '<br><br><strong>$1');
            
            messageContent.innerHTML = formattedMessage;
        } else {
        // For user messages
            const messageText = document.createElement('p');
            messageText.textContent = message;
            messageContent.appendChild(messageText);
        }
        
       
        const timestamp = document.createElement('div');
        timestamp.classList.add('message-timestamp');
        timestamp.textContent = getCurrentTime();
        messageContent.appendChild(timestamp);
        
        messageElement.appendChild(messageContent);
        chatMessages.appendChild(messageElement);
        
     
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

 
    function getCurrentTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    
    function fetchResponseFromBackend(userMessage) {
    
        showTypingIndicator();
        
        
        fetch('http://localhost:5000/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: userMessage }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            addMessage(data.response, 'bot');
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error connecting to the server. Please try again later.', 'bot');
        });
    }

   
    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'bot', 'typing-indicator');
        
        const indicatorContent = document.createElement('div');
        indicatorContent.classList.add('message-content');
        
        const dots = document.createElement('div');
        dots.classList.add('typing-dots');
        dots.innerHTML = '<span></span><span></span><span></span>';
        
        indicatorContent.appendChild(dots);
        typingIndicator.appendChild(indicatorContent);
        
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

  
    function removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }


    const style = document.createElement('style');
    style.textContent = `
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #aaa;
            animation: typing-dot 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing-dot {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.6;
            }
            30% {
                transform: translateY(-4px);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);
}); 