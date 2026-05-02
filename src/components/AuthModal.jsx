import React, { useState } from 'react';
import { useUser } from '../contexts/UserContext';
import './AuthModal.css';

const AuthModal = ({ isOpen, onClose, mode = 'login' }) => {
  const { login, signup, error: authError, isLoading } = useUser();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [formError, setFormError] = useState('');

  if (!isOpen) return null;

  const validateForm = () => {
    if (!username.trim()) {
      setFormError('Username is required');
      return false;
    }

    if (mode === 'signup') {
      if (!email.trim()) {
        setFormError('Email is required');
        return false;
      }

      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        setFormError('Please enter a valid email');
        return false;
      }
    }

    if (!password) {
      setFormError('Password is required');
      return false;
    }

    if (password.length < 6) {
      setFormError('Password must be at least 6 characters');
      return false;
    }

    if (mode === 'signup' && password !== confirmPassword) {
      setFormError('Passwords do not match');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    setFormError('');

    if (mode === 'login') {
      const result = await login(username, password);
      if (result.success) {
        onClose();
      } else {
        setFormError(result.error);
      }
    } else {
      const result = await signup(username, email, password);
      if (result.success) {
        onClose();
      } else {
        setFormError(result.error);
      }
    }
  };

  const switchMode = () => {
    setFormError('');
    // This will be handled by the parent component
  };

  return (
    <div className="auth-modal-overlay" onClick={onClose} role="presentation">
      <div 
        className="auth-modal" 
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
      >
        <div className="auth-modal-header">
          <h2 id="auth-modal-title">{mode === 'login' ? 'Login' : 'Sign Up'}</h2>
          <button className="auth-modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <form onSubmit={handleSubmit} className="auth-modal-form">
          {mode === 'signup' && (
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>

          {mode === 'signup' && (
            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                id="confirmPassword"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm your password"
                required
              />
            </div>
          )}

          {(formError || authError) && (
            <div className="auth-error" aria-live="assertive" role="alert">
              {formError || authError}
            </div>
          )}

          <button type="submit" className="auth-modal-submit" disabled={isLoading}>
            {isLoading ? 'Processing...' : mode === 'login' ? 'Login' : 'Sign Up'}
          </button>
        </form>

        <div className="auth-modal-footer">
          <p>
            {mode === 'login'
              ? "Don't have an account? "
              : "Already have an account? "}
            <button
              type="button"
              className="auth-modal-switch"
              onClick={switchMode}
            >
              {mode === 'login' ? 'Sign Up' : 'Login'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;