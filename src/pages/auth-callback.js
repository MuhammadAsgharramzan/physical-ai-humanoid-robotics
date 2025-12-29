import React, { useEffect } from 'react';
import Layout from '@theme/Layout';

function AuthCallback() {
  useEffect(() => {
    // This component simulates what would happen after OAuth redirect
    // In a real implementation, this would handle the OAuth callback
    const handleOAuthCallback = async () => {
      // Get the URL parameters (in a real app, these would come from the OAuth provider)
      const urlParams = new URLSearchParams(window.location.search);
      const provider = urlParams.get('provider') || 'unknown';
      const authType = urlParams.get('type') || 'login'; // 'login' or 'signup'
      const code = urlParams.get('code');

      // Show loading state
      document.title = 'Authenticating...';

      try {
        // Simulate exchanging the code for tokens
        // In a real implementation, this would be a call to your backend
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Create a mock user object similar to what OAuth providers return
        const mockUser = {
          id: `${provider.toLowerCase()}_${Date.now()}`,
          username: `${provider.toLowerCase()}_user_${Date.now()}`,
          email: `${provider.toLowerCase()}_user_${Date.now()}@${provider.toLowerCase()}.com`,
          provider: provider.toLowerCase(),
          name: `${provider} User`,
          avatar: null, // Would come from provider in real implementation
          verified: true
        };

        // Store the token and user info in localStorage
        const mockToken = `mock-${provider.toLowerCase()}-token-${Date.now()}`;
        localStorage.setItem('authToken', mockToken);
        localStorage.setItem('user', JSON.stringify(mockUser));

        // Show success message based on auth type
        if (authType === 'signup') {
          console.log(`${provider} signup successful! Welcome, ${mockUser.name}.`);
        } else {
          console.log(`${provider} login successful! Welcome back, ${mockUser.name}.`);
        }

        // Redirect to home page or the page the user was on
        window.location.href = '/';
      } catch (error) {
        console.error('OAuth callback error:', error);
        // Show error to user and redirect to login
        alert(`Authentication with ${provider} failed. Please try again.`);
        // Redirect to login with error message
        window.location.href = `/login?error=oauth_failed&provider=${provider}`;
      }
    };

    handleOAuthCallback();
  }, []);

  return (
    <Layout title="Authenticating..." description="Completing authentication process">
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
          Authenticating with {new URLSearchParams(window.location.search).get('provider') || 'provider'}...
        </div>
        <div style={{
          fontSize: '16px',
          color: '#666',
          marginBottom: '30px',
          textAlign: 'center'
        }}>
          {new URLSearchParams(window.location.search).get('type') === 'signup'
            ? 'Creating your account'
            : 'Logging you in'}
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
    </Layout>
  );
}

export default AuthCallback;