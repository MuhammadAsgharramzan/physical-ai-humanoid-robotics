import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { translate } from '@docusaurus/core/lib/client/exports/translate';
import BrowserOnly from '@docusaurus/BrowserOnly';

function AuthCallbackContent() {
  const [loading, setLoading] = useState(true);
  const [provider, setProvider] = useState('provider');
  const [authType, setAuthType] = useState('login');

  useEffect(() => {
    // This component simulates what would happen after OAuth redirect
    // In a real implementation, this would handle the OAuth callback
    const handleOAuthCallback = async () => {
      // Get the URL parameters (in a real app, these would come from the OAuth provider)
      const urlParams = new URLSearchParams(window.location.search);
      const currentProvider = urlParams.get('provider') || 'unknown';
      const currentAuthType = urlParams.get('type') || 'login'; // 'login' or 'signup'
      const code = urlParams.get('code');

      setProvider(currentProvider);
      setAuthType(currentAuthType);

      // Show loading state
      document.title = 'Authenticating...';

      try {
        // Simulate exchanging the code for tokens
        // In a real implementation, this would be a call to your backend
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Create a mock user object similar to what OAuth providers return
        const mockUser = {
          id: `${currentProvider.toLowerCase()}_${Date.now()}`,
          username: `${currentProvider.toLowerCase()}_user_${Date.now()}`,
          email: `${currentProvider.toLowerCase()}_user_${Date.now()}@${currentProvider.toLowerCase()}.com`,
          provider: currentProvider.toLowerCase(),
          name: `${currentProvider} User`,
          avatar: null, // Would come from provider in real implementation
          verified: true
        };

        // Store the token and user info in localStorage
        const mockToken = `mock-${currentProvider.toLowerCase()}-token-${Date.now()}`;
        localStorage.setItem('authToken', mockToken);
        localStorage.setItem('user', JSON.stringify(mockUser));

        // Show success message based on auth type
        if (currentAuthType === 'signup') {
          console.log(translate({
            id: 'authentication.success.social_signup',
            message: `{provider} signup successful! Welcome, {name}.`,
            values: { provider: currentProvider, name: mockUser.name }
          }));
        } else {
          console.log(translate({
            id: 'authentication.success.social_login',
            message: `{provider} login successful! Welcome back, {name}.`,
            values: { provider: currentProvider, name: mockUser.name }
          }));
        }

        // Redirect to home page or the page the user was on
        window.location.href = '/';
      } catch (error) {
        console.error('OAuth callback error:', error);
        // Show error to user and redirect to login
        alert(translate({
          id: 'authentication.error.oauth_failed',
          message: `Authentication with {provider} failed. Please try again.`,
          values: { provider: currentProvider }
        }));
        // Redirect to login with error message
        window.location.href = `/login?error=oauth_failed&provider=${currentProvider}`;
      }
    };

    handleOAuthCallback();
  }, []);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '70vh',
      flexDirection: 'column'
    }}>
      <div style={{
        fontSize: '24px',
        marginBottom: '20px',
        textAlign: 'center'
      }}>
        {translate({
          id: 'authentication.authcallback.header',
          message: 'Authenticating with {provider}...',
          values: { provider: provider }
        })}
      </div>
      <div style={{
        fontSize: '16px',
        color: '#666',
        marginBottom: '30px',
        textAlign: 'center'
      }}>
        {authType === 'signup'
          ? translate({ id: 'authentication.authcallback.creating_account', message: 'Creating your account' })
          : translate({ id: 'authentication.authcallback.logging_in', message: 'Logging you in' })}
      </div>
      <div className="loading-spinner" style={{
        width: '40px',
        height: '40px',
        border: '4px solid #f3f3f3',
        borderTop: '4px solid #3498db',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite'
      }}></div>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

function AuthCallback() {
  return (
    <Layout
      title={translate({ id: 'authentication.authcallback.title', message: 'Authenticating...' })}
      description={translate({ id: 'authentication.authcallback.description', message: 'Completing authentication process' })}>
      <BrowserOnly>
        {() => <AuthCallbackContent />}
      </BrowserOnly>
    </Layout>
  );
}

export default AuthCallback;