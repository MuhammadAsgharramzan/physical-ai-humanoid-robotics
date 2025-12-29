import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotWidget from '@site/src/components/ChatbotWidget';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        {/* Chatbot widget */}
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: '1000',
        }}>
          <ChatbotWidget />
        </div>
      </OriginalLayout>
    </>
  );
}