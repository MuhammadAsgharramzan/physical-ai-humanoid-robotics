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

  // Simple approach: construct the path manually
  const currentPath = location.pathname;
  let targetPath = currentPath;

  // Remove current locale from path if present
  if (currentPath.startsWith(`/${currentLocale}/`)) {
    targetPath = currentPath.substring(currentLocale.length + 1);
  } else if (currentPath === `/${currentLocale}`) {
    targetPath = '/';
  }

  // Add target locale to path if it's not the default
  if (targetLocale !== i18n.defaultLocale) {
    if (targetPath === '/') {
      targetPath = `/${targetLocale}`;
    } else {
      targetPath = `/${targetLocale}${targetPath}`;
    }
  }

  const toggleLanguage = () => {
    // Navigate to the new locale path
    window.location.href = targetPath;
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