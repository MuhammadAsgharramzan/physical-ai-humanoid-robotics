import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css';

const Chatbot = ({ apiUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  // Initialize conversation
  useEffect(() => {
    // Try to load conversation from localStorage
    const savedConversation = localStorage.getItem('chatbot-conversation');
    if (savedConversation) {
      const conversation = JSON.parse(savedConversation);
      setConversationId(conversation.id);
      setMessages(conversation.messages);
    } else {
      // Generate a unique conversation ID
      const id = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setConversationId(id);

      // Add welcome message
      setMessages([
        {
          id: 'welcome',
          role: 'assistant',
          content: 'Hello! I\'m your Physical AI & Humanoid Robotics assistant. How can I help you with the textbook today?',
          timestamp: new Date()
        }
      ]);
    }
  }, []);

  // Save conversation to localStorage whenever messages change
  useEffect(() => {
    if (conversationId && messages.length > 0) {
      const conversation = {
        id: conversationId,
        messages: messages,
        timestamp: new Date().toISOString()
      };
      localStorage.setItem('chatbot-conversation', JSON.stringify(conversation));
    }
  }, [messages, conversationId]);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API
      const response = await fetch(`${apiUrl}/chat?conversation_id=${conversationId}&user_message=${encodeURIComponent(inputValue)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: `bot_${Date.now()}`,
          role: 'assistant',
          content: data.response,
          citations: data.citations,
          confidence: data.confidence,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorData = await response.json();
        const errorMessage = {
          id: `error_${Date.now()}`,
          role: 'assistant',
          content: `Sorry, I encountered an error: ${errorData.detail || 'Unknown error'}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I\'m having trouble connecting to the server. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatMessage = (content) => {
    // Simple formatting for citations
    return content.split(/\[Source \d+\]/).map((part, index, array) => {
      const sourceMatch = content.match(new RegExp(`\\[Source ${index + 1}\\]`, 'g'));
      return (
        <React.Fragment key={index}>
          {part}
          {sourceMatch && (
            <sup className="citation">[{index + 1}]</sup>
          )}
        </React.Fragment>
      );
    });
  };

  return (
    <div className="chatbot-container" role="complementary" aria-label="AI Chatbot Assistant">
      <div className="chatbot-header">
        <h3>Physical AI & Humanoid Robotics Assistant</h3>
      </div>

      <div
        className="chatbot-messages"
        aria-live="polite"
        aria-relevant="additions"
        role="log"
        aria-label="Chat messages"
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.role}`}
            role="listitem"
          >
            <div className="message-header">
              <strong>{message.role === 'user' ? 'You' : 'Assistant'}</strong>
              <span className="timestamp" aria-label={`Sent at ${message.timestamp.toLocaleTimeString()}`}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            <div className="message-content">
              {formatMessage(message.content)}
              {message.citations && message.citations.length > 0 && (
                <div className="citations" aria-label="Sources cited in this response">
                  <h4>Sources:</h4>
                  <ul>
                    {message.citations.map((citation, idx) => (
                      <li key={idx}>
                        {citation.title} (Module: {citation.module_id}, Lesson: {citation.lesson_id})
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant" role="status" aria-label="Assistant is typing">
            <div className="message-header">
              <strong>Assistant</strong>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chatbot-input-form" role="form">
        <div className="input-container">
          <label htmlFor="chatbot-input" className="sr-only">Type your message</label>
          <input
            id="chatbot-input"
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about Physical AI, robotics, or the textbook content..."
            disabled={isLoading}
            aria-label="Type your message to the chatbot"
            autoComplete="off"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            aria-label="Send message"
            className="send-button"
          >
            Send
          </button>
        </div>
        <div className="input-hints" aria-label="Usage hints">
          <small>Ask questions about the textbook content, concepts, or examples.</small>
        </div>
      </form>
    </div>
  );
};

export default Chatbot;