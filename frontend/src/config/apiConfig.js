// API Configuration
// Use environment variable if available, otherwise default based on runtime environment
const getApiUrl = () => {
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // In the browser, automatically use current origin (same-origin serving on Render)
  if (typeof window !== 'undefined' && window.location) {
    if (window.location.port === '3000') {
      return 'http://localhost:7860';
    }
    return window.location.origin;
  }
  
  return '';
};

export const API_URL = getApiUrl();

export default API_URL;

