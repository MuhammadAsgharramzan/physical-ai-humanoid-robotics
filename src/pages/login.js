import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useUser } from '../contexts/UserContext';
import { translate } from '@docusaurus/core/lib/client/exports/translate';
import './AuthPage.css';

function LoginPageContent() {
  const { login, error: authError, isLoading } = useUser();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [formError, setFormError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!username.trim() || !password) {
      setFormError(translate({ id: 'authentication.validation.both_fields_required', message: 'Both fields are required' }));
      return;
    }

    setFormError('');

    // Call the login function from UserContext
    const result = await login(username, password);
    if (result.success) {
      // Successful login - redirect to home
      window.location.href = '/';
    } else {
      setFormError(result.error || translate({ id: 'authentication.error.login_failed', message: 'Login failed' }));
    }
  };

  const handleSocialLogin = async (provider) => {
    // Show loading state
    setFormError('');

    try {
      // Simulate OAuth redirect process
      console.log(`Initiating ${provider} login...`);

      // Show loading indication
      const socialButton = document.querySelector(`.social-login-button.${provider.toLowerCase()}`);
      if (socialButton) {
        const originalText = socialButton.innerHTML;
        socialButton.innerHTML = translate({
          id: 'authentication.button.redirecting_to',
          message: `Redirecting to ${provider}...`,
          values: { provider }
        });
        socialButton.disabled = true;

        // In a real implementation, this would redirect to the OAuth provider
        // For this mock, we'll redirect to our auth callback page to simulate the flow
        setTimeout(() => {
          // Restore button state
          if (socialButton) {
            socialButton.innerHTML = originalText;
            socialButton.disabled = false;
          }

          // Redirect to auth callback page to simulate OAuth flow
          window.location.href = `/auth-callback?provider=${provider}&type=login`;
        }, 1500);
      }
    } catch (error) {
      console.error(`Error during ${provider} login:`, error);
      setFormError(translate({
        id: 'authentication.error.social_login_failed',
        message: `Failed to login with ${provider}. Please try again.`,
        values: { provider }
      }));

      // Restore button state
      const socialButton = document.querySelector(`.social-login-button.${provider.toLowerCase()}`);
      if (socialButton) {
        const originalText = socialButton.innerHTML;
        socialButton.innerHTML = originalText;
        socialButton.disabled = false;
      }
    }
  };

  return (
    <div className="auth-page-container">
      <div className="auth-form-wrapper">
        <h1>{translate({ id: 'authentication.login.header', message: 'Login to Your Account' })}</h1>

        {/* Social Login Options */}
        <div className="social-login-section">
          <div className="social-login-divider">
            <span>{translate({ id: 'authentication.divider.or_continue_with', message: 'Or continue with' })}</span>
          </div>
          <div className="social-login-buttons">
            <button className="social-login-button google" onClick={() => handleSocialLogin('Google')}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22.56 12.25C22.56 11.47 22.49 10.72 22.36 10H12V14.23H17.84C17.57 15.67 16.61 16.85 15.28 17.58V20.35H18.8C20.83 18.46 22.09 15.73 22.56 12.25Z" fill="#4285F4"/>
                <path d="M12 23C14.97 23 17.46 22.02 19.37 20.25L15.28 17.58C14.21 18.31 12.93 18.74 12 18.74C9.13 18.74 6.7 16.31 6.7 13.44C6.7 10.57 9.13 8.14 12 8.14C13.4 8.14 14.64 8.65 15.6 9.5L18.58 6.52C16.77 4.78 14.5 3.75 12 3.75C7.63 3.75 3.75 7.63 3.75 12C3.75 16.37 7.63 20.25 12 20.25C15.39 20.25 18.14 17.83 19.48 14.63L16.06 12.25H12Z" fill="#34A853"/>
                <path d="M6.7 13.44C6.54 12.94 6.54 12.42 6.7 11.92V9.5H3.75C3.09 10.89 2.75 12.42 2.75 14C2.75 15.58 3.09 17.11 3.75 18.5C5.09 21.19 7.63 23 12 23C14.5 23 16.77 21.97 18.58 20.23L15.28 17.58C14.48 18.19 13.4 18.58 12 18.58C9.13 18.58 6.7 16.15 6.7 13.28V13.44Z" fill="#FBBC05"/>
                <path d="M12 5.75C13.3 5.75 14.52 6.15 15.5 7L18.58 4C16.77 2.21 14.5 1 12 1C7.63 1 3.75 4.88 3.75 9.25C3.75 10.83 4.09 12.36 4.75 13.75C5.41 15.14 6.33 16.31 7.41 17.19L10.91 14.45C10.21 13.95 9.75 13.21 9.75 12.44C9.75 11.07 10.62 9.94 11.75 9.5L12 9.25C11.21 8.5 10.75 7.5 10.75 6.5C10.75 6.08 10.86 5.69 11 5.34L12 5.75Z" fill="#EA4335"/>
              </svg>
              {translate({ id: 'authentication.button.continue_with_google', message: 'Continue with Google' })}
            </button>
            <button className="social-login-button github" onClick={() => handleSocialLogin('GitHub')}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 0C5.37 0 0 5.37 0 12C0 17.3 3.43 21.8 8.2 23.38C8.8 23.49 9 23.13 9 22.82V20.5C6.07 21.13 5.2 19.17 5.2 19.17C4.6 17.64 3.74 17.2 3.74 17.2C2.56 16.38 3.82 16.39 3.82 16.39C5.1 16.48 5.8 17.63 5.8 17.63C6.96 19.44 8.68 18.97 9.06 18.78C9.12 18.17 9.37 17.7 9.64 17.45C7.5 17.22 5.3 16.38 5.3 12.34C5.3 11.05 5.76 9.99 6.55 9.17C6.42 8.86 6 7.95 6.7 6.1C6.7 6.1 7.57 5.83 9 6.95C9.63 6.78 10.32 6.7 11 6.7C11.68 6.7 12.37 6.78 13 6.95C14.43 5.82 15.3 6.1 15.3 6.1C16 7.95 15.58 8.86 15.45 9.17C16.24 9.99 16.7 11.05 16.7 12.34C16.7 16.39 14.5 17.2 12.4 17.44C12.7 17.7 13 18.16 13 18.82V22.82C13 23.13 13.2 23.49 13.8 23.38C18.57 21.8 22 17.3 22 12C22 5.37 16.63 0 12 0Z"/>
              </svg>
              {translate({ id: 'authentication.button.continue_with_github', message: 'Continue with GitHub' })}
            </button>
            <button className="social-login-button apple" onClick={() => handleSocialLogin('Apple')}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                <path d="M17.05 12.04C17.08 9.27 19.44 7.64 19.56 7.56C18.24 5.68 16.3 5.08 15.36 5.04C13.63 4.88 11.88 6.08 10.9 6.08C9.92 6.08 8.72 5.15 7.24 5.2C5.68 5.25 4.16 6.4 3 8.04C1.04 10.92 2.2 15.44 4.2 18.12C5.36 19.68 6.76 21.48 8.44 21.44C9.96 21.44 10.8 20.48 12.6 20.48C14.4 20.48 15.16 21.44 16.84 21.44C18.64 21.44 19.72 19.76 20.84 18.2C21.64 17.08 22.12 15.84 22.08 14.6C21.96 12.92 20.2 11.75 18.72 11.72C17.96 11.72 17.04 12.04 17.05 12.04ZM14.84 4.24C15.64 3.16 16.2 1.64 16.04 0C14.56 0.04 12.96 1.04 12.16 2.12C11.36 3.2 10.8 4.76 10.96 6.44C12.44 6.52 13.96 5.48 14.84 4.24Z"/>
              </svg>
              {translate({ id: 'authentication.button.continue_with_apple', message: 'Continue with Apple' })}
            </button>
          </div>
        </div>

        {/* Divider */}
        <div className="social-login-divider">
          <span>{translate({ id: 'authentication.divider.or_with_email', message: 'Or with email' })}</span>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="username">{translate({ id: 'authentication.field.username', message: 'Username' })}</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder={translate({ id: 'authentication.placeholder.username', message: 'Enter your username' })}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">{translate({ id: 'authentication.field.password', message: 'Password' })}</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={translate({ id: 'authentication.placeholder.enter_password', message: 'Enter your password' })}
              required
            />
          </div>

          {(formError || authError) && (
            <div className="auth-error">
              {formError || authError}
            </div>
          )}

          <button type="submit" className="auth-submit-button" disabled={isLoading}>
            {isLoading
              ? translate({ id: 'authentication.button.logging_in', message: 'Logging in...' })
              : translate({ id: 'authentication.button.login', message: 'Login' })}
          </button>
        </form>

        <div className="auth-page-footer">
          <p>
            {translate({ id: 'authentication.footer.dont_have_account', message: "Don't have an account?" })}{' '}
            <a href="/signup">{translate({ id: 'authentication.footer.signup_link', message: 'Sign up' })}</a>
          </p>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Layout
      title={translate({ id: 'authentication.login.title', message: 'Login' })}
      description={translate({ id: 'authentication.login.description', message: 'Login to your account' })}>
      <LoginPageContent />
    </Layout>
  );
}