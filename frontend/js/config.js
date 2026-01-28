/**
 * Frontend Configuration
 * Centralized API and application settings
 */

const CONFIG = {
    // API Configuration
    API_BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:5000'
        : 'https://your-backend-url.onrender.com', // Update with deployed URL
    
    // Request Settings
    REQUEST_TIMEOUT: 30000, // 30 seconds
    REQUEST_HEADERS: {
        'Content-Type': 'application/json'
    },
    
    // UI Settings
    MAX_MESSAGE_LENGTH: 2000,
    TYPING_INDICATOR_DELAY: 500, // ms
    AUTO_SCROLL_DELAY: 100, // ms
    
    // Session Management
    SESSION_STORAGE_KEY: 'rag_chatbot_session_id',
    
    // Feature Flags
    ENABLE_MARKDOWN: false,
    ENABLE_CODE_HIGHLIGHTING: false,
    DEBUG_MODE: false
};

// Generate or retrieve session ID
export const API_BASE_URL = window.location.origin;
export function getSessionId() {
  let sessionId = localStorage.getItem(CONFIG.SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem(CONFIG.SESSION_STORAGE_KEY, sessionId);
  }
  return sessionId;
}

// NEW: Export CONFIG for ES modules
export { CONFIG };

// NEW: Utility helpers
export function isValidGoogleDocUrl(url) {
  return /^https?:\/\/docs\.google\.com\/document\/d\/[a-zA-Z0-9_-]+/.test(url);
}
export function debugLog(...args) {
  // Respect debug flag if needed
  if (CONFIG.DEBUG_MODE) {
    console.log('[RAG]', ...args);
  } else {
    // Always allow key startup logs
    if (args[0] && typeof args[0] === 'string' && args[0].includes('Backend health OK')) {
      console.log('[RAG]', ...args);
    }
  }
}

// Export for use in other modules (CommonJS fallback)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, API_BASE_URL, getSessionId, isValidGoogleDocUrl, debugLog };
}