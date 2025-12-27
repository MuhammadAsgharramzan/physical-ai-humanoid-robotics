import React, { useState, useEffect } from 'react';
import Chatbot from './Chatbot/Chatbot';
import './Chatbot/Chatbot.css';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);

  // Only load the chatbot component after user interaction to improve performance
  const loadChatbot = () => {
    if (!hasLoaded) {
      setHasLoaded(true);
    }
    setIsOpen(true);
  };

  return (
    <div className="chatbot-widget">
      {isOpen ? (
        <div className="chatbot-panel">
          <div className="chatbot-header">
            <h3>Physical AI Assistant</h3>
            <button
              className="close-button"
              onClick={() => setIsOpen(false)}
              aria-label="Close chatbot"
            >
              ×
            </button>
          </div>
          {hasLoaded && <Chatbot />}
        </div>
      ) : (
        <button
          className="chatbot-toggle-button"
          onClick={loadChatbot}
          aria-label="Open chatbot assistant"
          aria-expanded="false"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;