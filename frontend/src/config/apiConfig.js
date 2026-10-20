// API Configuration
// Dynamically resolves to current origin in browser (Render / Cloud / local)
const getApiUrl = () => {
  // If explicitly provided via custom env (and NOT localhost)
  if (process.env.REACT_APP_API_URL && !process.env.REACT_APP_API_URL.includes('localhost:7860')) {
    return process.env.REACT_APP_API_URL;
  }
  
  // In the browser, always use the current window location
  if (typeof window !== 'undefined' && window.location) {
    // Only if running local React development server on port 3000
    if (window.location.hostname === 'localhost' && window.location.port === '3000') {
      return 'http://localhost:7860';
    }
    // In production on Render, HF Spaces, AWS, or any remote client:
    return window.location.origin;
  }
  
  return '';
};

export const API_URL = getApiUrl();

export default API_URL;


