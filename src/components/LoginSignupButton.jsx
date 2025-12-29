import React, { useState } from 'react';
import { useUser } from '../contexts/UserContext';
import UserProfile from './UserProfile';
import AuthModal from './AuthModal';
import './LoginSignupButton.css';

const LoginSignupButton = () => {
  const { isAuthenticated } = useUser();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState('login'); // 'login' or 'signup'

  const handleLoginClick = () => {
    setModalMode('login');
    setIsModalOpen(true);
  };

  const handleSignupClick = () => {
    setModalMode('signup');
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  if (isAuthenticated) {
    return <UserProfile />;
  }

  return (
    <>
      <div className="login-signup-container">
        <button
          className="auth-button login-button"
          onClick={handleLoginClick}
          aria-label="Login"
        >
          Login
        </button>
        <button
          className="auth-button signup-button"
          onClick={handleSignupClick}
          aria-label="Sign up"
        >
          Sign Up
        </button>
      </div>
      <AuthModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        mode={modalMode}
      />
    </>
  );
};

export default LoginSignupButton;