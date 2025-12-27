import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import TranslateButton from '@site/src/theme/components/TranslateButton';
import ChatbotWidget from '@site/src/components/ChatbotWidget';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
        {/* Floating translate button in the bottom right corner */}
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: '1000',
        }}>
          <TranslateButton isFloating={true} />
        </div>
        {/* Chatbot widget */}
        <div style={{
          position: 'fixed',
          bottom: '90px', // Position above the translate button
          right: '20px',
          zIndex: '1000',
        }}>
          <ChatbotWidget />
        </div>
      </OriginalLayout>
    </>
  );
}