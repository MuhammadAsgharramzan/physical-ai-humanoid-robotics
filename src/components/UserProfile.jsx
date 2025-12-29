import React from 'react';
import { useUser } from '../contexts/UserContext';
import './UserProfile.css';

const UserProfile = () => {
  const { user, isAuthenticated, logout } = useUser();

  if (!isAuthenticated || !user) return null;

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="user-profile">
      <div className="user-profile-menu">
        <div className="user-profile-info">
          <div className="user-avatar">
            {user.username?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <span className="user-name">{user.username}</span>
        </div>
        <button className="logout-button" onClick={handleLogout}>
          Logout
        </button>
      </div>
    </div>
  );
};

export default UserProfile;