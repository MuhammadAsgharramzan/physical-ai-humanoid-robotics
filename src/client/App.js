import React from 'react';
import { UserProvider } from './src/contexts/UserContext';

export default function App({ children }) {
  return <UserProvider>{children}</UserProvider>;
}