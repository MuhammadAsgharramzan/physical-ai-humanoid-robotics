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

      // Convert timestamp strings back to Date objects
      const messagesWithDates = conversation.messages.map(message => ({
        ...message,
        timestamp: new Date(message.timestamp)
      }));

      setMessages(messagesWithDates);
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
          timestamp: new Date().toISOString()
        }
      ]);
    }
  }, []);

  // Save conversation to localStorage whenever messages change
  useEffect(() => {
    if (conversationId && messages.length > 0) {
      // Convert Date objects to ISO strings for storage
      const messagesForStorage = messages.map(message => ({
        ...message,
        timestamp: message.timestamp instanceof Date ? message.timestamp.toISOString() : message.timestamp
      }));

      const conversation = {
        id: conversationId,
        messages: messagesForStorage,
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
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      // Try to call the backend API
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
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        // If API is not available, use mock responses
        throw new Error('API not available');
      }
    } catch (error) {
      // Generate mock response based on user input
      const mockResponses = [
        "I'm your Physical AI & Humanoid Robotics assistant. I can help explain concepts about embodied agents, sensorimotor integration, and AI techniques for robotics.",
        "Physical AI combines digital intelligence with real-world interaction through embodied agents. This field explores how machines can learn and interact with the physical world.",
        "Humanoid robotics involves creating robots with human-like characteristics and behaviors. These robots can interact with human environments and perform tasks similar to humans.",
        "Embodied intelligence refers to intelligence that emerges from the interaction between an agent and its environment. It's a key concept in physical AI.",
        "The textbook covers topics like computer vision for robotics, path planning, human-robot interaction, and advanced control systems.",
        "For practical implementation, you'll learn about ROS2, Gazebo simulation, and Isaac Sim for developing and testing robotic systems.",
        "The RAG system would normally search the textbook content to provide specific answers to your questions based on the course material."
      ];

      // Simple response generation based on keywords
      let responseContent = "I understand you're asking about Physical AI and Humanoid Robotics. In a live environment, I would search the textbook content to provide you with specific information and citations. Since the backend is not running, I'm providing a general response based on the course topics.";

      if (inputValue.toLowerCase().includes('hi') || inputValue.toLowerCase().includes('hello')) {
        responseContent = "Hello! I'm your Physical AI & Humanoid Robotics assistant. I can help explain concepts about embodied agents, sensorimotor integration, and AI techniques for robotics. What would you like to learn about?";
      } else if (inputValue.toLowerCase().includes('physical ai') || inputValue.toLowerCase().includes('embodied')) {
        responseContent = "Physical AI combines digital intelligence with real-world interaction through embodied agents. It's a key field that explores how machines can learn and interact with the physical world, which is different from traditional AI that operates purely in digital spaces.";
      } else if (inputValue.toLowerCase().includes('robot') || inputValue.toLowerCase().includes('humanoid')) {
        responseContent = "Humanoid robotics involves creating robots with human-like characteristics and behaviors. These robots are designed to interact with human environments and perform tasks similar to humans. The field combines mechanical engineering, AI, and human factors design.";
      } else if (inputValue.toLowerCase().includes('chat') || inputValue.toLowerCase().includes('help')) {
        responseContent = "I'm here to help you learn about Physical AI & Humanoid Robotics! You can ask me about concepts from the textbook, such as embodied intelligence, sensorimotor integration, computer vision for robotics, path planning, or human-robot interaction. In a live environment, I would search the full textbook content to provide specific answers.";
      } else {
        // Pick a random response from the mock responses
        responseContent = mockResponses[Math.floor(Math.random() * mockResponses.length)];
      }

      const botMessage = {
        id: `bot_${Date.now()}`,
        role: 'assistant',
        content: responseContent,
        citations: [{title: "Physical AI & Humanoid Robotics Textbook", module_id: "intro", lesson_id: "overview"}],
        confidence: 0.8,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, botMessage]);
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
              <span className="timestamp" aria-label={`Sent at ${new Date(message.timestamp).toLocaleTimeString()}`}>
                {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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