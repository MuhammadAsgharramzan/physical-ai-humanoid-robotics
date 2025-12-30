import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Translate from '@docusaurus/Translate';

const TranslateButton = ({ isFloating = false }) => {
  const location = useLocation();
  const { i18n } = useDocusaurusContext();
  const [currentLocale, setCurrentLocale] = useState('');

  useEffect(() => {
    if (i18n && i18n.currentLocale) {
      setCurrentLocale(i18n.currentLocale);
    }
  }, [i18n]);

  if (!i18n || !i18n.locales || i18n.locales.length <= 1) {
    return null;
  }

  const isUrdu = currentLocale === 'ur';
  const targetLocale = isUrdu ? 'en' : 'ur';

  const toggleLanguage = () => {
    // Get the current path without the locale prefix
    let currentPath = location.pathname;

    // Remove current locale from path if present
    if (currentPath.startsWith(`/${currentLocale}/`)) {
      currentPath = currentPath.substring(currentLocale.length + 1);
    } else if (currentPath === `/${currentLocale}`) {
      currentPath = '/';
    } else if (currentPath.startsWith('/en/') || currentPath.startsWith('/ur/')) {
      // Remove any existing locale prefix
      const pathParts = currentPath.split('/');
      if (pathParts.length > 1) {
        currentPath = '/' + pathParts.slice(2).join('/');
        if (!currentPath.startsWith('/')) currentPath = '/' + currentPath;
      }
    }

    // Construct the new path with the target locale
    let newPath;
    if (targetLocale === i18n.defaultLocale) {
      // For default locale (English), we typically don't include the locale in the path
      newPath = currentPath;
    } else {
      // For non-default locales (Urdu), include the locale prefix
      if (currentPath === '/') {
        newPath = `/${targetLocale}`;
      } else {
        newPath = `/${targetLocale}${currentPath}`;
      }
    }

    // Navigate to the new locale path using a full page navigation
    // This ensures Docusaurus properly handles the locale switching
    window.location.href = `${newPath}${location.search}${location.hash}`;
  };

  const buttonStyle = {
    backgroundColor: isUrdu ? '#d2691e' : '#4285f4', // Brown for Urdu, Blue for English
    color: 'white',
    border: 'none',
    padding: isFloating ? '12px 20px' : '8px 16px',
    borderRadius: '25px',
    cursor: 'pointer',
    fontSize: isFloating ? '16px' : '14px',
    fontWeight: '600',
    textTransform: 'uppercase',
    boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
    transition: 'all 0.3s ease',
    fontFamily: isUrdu ? 'Tahoma, Arial, sans-serif' : 'inherit',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',
  };

  if (!isFloating) {
    buttonStyle.marginLeft = '10px';
  } else {
    buttonStyle.position = 'relative';
    buttonStyle.zIndex = '1000';
  }

  const flagEmoji = isUrdu ? '🇵🇰' : '🇬🇧'; // Pakistan flag for Urdu, UK flag for English

  return (
    <button
      onClick={toggleLanguage}
      className={`translate-button ${isFloating ? 'floating-translate-button' : ''}`}
      style={buttonStyle}
      title={isUrdu ? 'Switch to English' : 'اردو میں تبدیل کریں'}
      aria-label={isUrdu ? 'Switch to English' : 'اردو میں تبدیل کریں'}
    >
      <span>{flagEmoji}</span>
      <span>{isUrdu ? 'EN' : 'UR'}</span>
    </button>
  );
};

export default TranslateButton;