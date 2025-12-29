import React, { createContext, useContext, useReducer, useEffect } from 'react';

const UserContext = createContext();

const userReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        user: action.payload,
        isAuthenticated: true,
        isLoading: false
      };
    case 'LOGOUT':
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        isLoading: false
      };
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload
      };
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false
      };
    default:
      return state;
  }
};

export const UserProvider = ({ children }) => {
  const [state, dispatch] = useReducer(userReducer, {
    user: null,
    isAuthenticated: false,
    isLoading: true,
    error: null
  });

  useEffect(() => {
    // Check if user is logged in on initial load
    const token = localStorage.getItem('authToken');
    const storedUser = localStorage.getItem('user');

    if (storedUser) {
      // If we have a stored user object (from social login), use that
      try {
        const userData = JSON.parse(storedUser);
        dispatch({
          type: 'LOGIN_SUCCESS',
          payload: userData
        });
      } catch (error) {
        console.error('Error parsing stored user data:', error);
        localStorage.removeItem('user');
        dispatch({ type: 'LOGOUT' });
      }
    } else if (token) {
      // If we have a JWT token, decode it
      try {
        // Decode JWT to get user info
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(
          atob(base64)
            .split('')
            .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
            .join('')
        );

        const userData = JSON.parse(jsonPayload);
        dispatch({
          type: 'LOGIN_SUCCESS',
          payload: { username: userData.sub, email: userData.email }
        });
      } catch (error) {
        console.error('Error decoding token:', error);
        localStorage.removeItem('authToken');
        dispatch({ type: 'LOGOUT' });
      }
    } else {
      dispatch({ type: 'LOGOUT' });
    }
  }, []);

  const login = async (username, password) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });

    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (response.ok) {
        const authToken = data.access_token;
        const userData = { username: data.user.username, email: data.user.email, id: data.user.id };

        localStorage.setItem('authToken', authToken);
        localStorage.setItem('user', JSON.stringify(userData));

        dispatch({
          type: 'LOGIN_SUCCESS',
          payload: userData
        });
        return { success: true };
      } else {
        dispatch({
          type: 'SET_ERROR',
          payload: data.detail || 'Login failed'
        });
        return { success: false, error: data.detail || 'Login failed' };
      }
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        payload: 'Network error. Please try again.'
      });
      return { success: false, error: 'Network error. Please try again.' };
    }
  };

  const signup = async (username, email, password) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });

    try {
      const response = await fetch('http://localhost:8000/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        const authToken = data.access_token;
        const userData = { username: data.username, email: data.email, id: data.id };

        localStorage.setItem('authToken', authToken);
        localStorage.setItem('user', JSON.stringify(userData));

        dispatch({
          type: 'LOGIN_SUCCESS',
          payload: userData
        });
        return { success: true };
      } else {
        dispatch({
          type: 'SET_ERROR',
          payload: data.detail || 'Registration failed'
        });
        return { success: false, error: data.detail || 'Registration failed' };
      }
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        payload: 'Network error. Please try again.'
      });
      return { success: false, error: 'Network error. Please try again.' };
    }
  };

  const logout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    dispatch({ type: 'LOGOUT' });
  };

  return (
    <UserContext.Provider
      value={{
        ...state,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};