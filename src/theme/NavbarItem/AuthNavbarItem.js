import React from 'react';
import NavbarItem from '@theme/NavbarItem';
import LoginSignupButton from '@site/src/components/LoginSignupButton';

function AuthNavbarItem() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
      <LoginSignupButton />
    </div>
  );
}

export default AuthNavbarItem;