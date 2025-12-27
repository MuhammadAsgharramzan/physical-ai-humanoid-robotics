import { useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import CustomFooter from '@site/src/components/CustomFooter';

function FooterInjector() {
  useEffect(() => {
    if (!ExecutionEnvironment.canUseDOM) {
      return undefined;
    }

    const footerContainer = document.createElement('div');
    footerContainer.id = 'custom-footer-container';
    
    // Insert the footer at the end of the body
    document.body.appendChild(footerContainer);

    // Render the custom footer
    const renderFooter = () => {
      // Remove any existing custom footer
      const existingFooter = document.getElementById('custom-footer-container');
      if (existingFooter) {
        existingFooter.remove();
      }
      
      // Create new container
      const newFooterContainer = document.createElement('div');
      newFooterContainer.id = 'custom-footer-container';
      document.body.appendChild(newFooterContainer);
      
      // This would normally use ReactDOM.createRoot for React 18
      // For Docusaurus compatibility, we'll just render it as part of the layout instead
    };

    renderFooter();

    return () => {
      const footerContainer = document.getElementById('custom-footer-container');
      if (footerContainer) {
        footerContainer.remove();
      }
    };
  }, []);

  return null;
}

export default FooterInjector;