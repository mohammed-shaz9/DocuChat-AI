import { CONFIG, getSessionId, isValidGoogleDocUrl, debugLog } from './config.js';
import { checkHealth, ingestDocument, sendChat, clearConversation } from './api.js';
import { handleSendMessage, handleClearChat } from './chatManager.js';
import { initializeUI, showLoading, hideLoading, updateStatus, showFeedback, switchToChat, appendMessage, showExampleQuestions, showDocStats } from './uiController.js';

const sessionId = getSessionId();

// STEP 3: Initialize the app
async function initializeApp() {
  console.log('Initializing RAG Chatbot...');
  console.log('Session ID:', sessionId);

  // Initialize UI elements
  initializeUI();

  // Health check
  try {
    const health = await checkHealth();
    updateStatus(true, health);
    debugLog('Backend health OK:', health);
  } catch (err) {
    updateStatus(false);
    showFeedback(`Backend not reachable: ${err.message}`, 'error');
  }

  // Wire event listeners
  attachEventListeners();
  console.log('App initialized successfully');
}

// STEP 4: Attach event listeners
function attachEventListeners() {
  const loadDocBtn = document.getElementById('load-doc-btn');
  if (loadDocBtn) loadDocBtn.addEventListener('click', handleLoadDocument);

  const sendBtn = document.getElementById('send-btn');
  if (sendBtn) sendBtn.addEventListener('click', handleSendMessage);

  const clearBtn = document.getElementById('clear-chat-btn');
  if (clearBtn) clearBtn.addEventListener('click', handleClearChat);

  const userInput = document.getElementById('user-input');
  if (userInput) {
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
  }

  const docUrlInput = document.getElementById('doc-url-input');
  if (docUrlInput) {
    docUrlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleLoadDocument();
      }
    });
  }

  debugLog('Event listeners attached');

  // Handle custom chat events
  window.addEventListener('rag:sendMessage', async (e) => {
    const text = (e?.detail?.text || '').trim();
    if (!text) return;
    appendMessage({ role: 'user', text });
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.style.display = 'flex';
    try {
      const resp = await sendChat(text, sessionId);
      const answer = resp?.answer || 'No answer available.';
      appendMessage({
        role: 'assistant',
        text: answer,
        confidence: resp.confidence,
        sources: resp.sources
      });
    } catch (err) {
      appendMessage({ role: 'assistant', text: `Error: ${err.message}` });
    } finally {
      if (typing) typing.style.display = 'none';
      const input = document.getElementById('user-input');
      if (input) input.value = '';
    }
  });

  window.addEventListener('rag:clearChat', async () => {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.style.display = 'none';
    const container = document.getElementById('chat-messages');
    if (container) container.innerHTML = '';
    try {
      await clearConversation(sessionId);
      showFeedback('Conversation cleared.', 'info');
      // Re-show example questions
      showExampleQuestions([
        "What is the main topic?",
        "Summarize the key points",
        "What are the conclusions?"
      ], (t) => window.dispatchEvent(new CustomEvent('rag:sendMessage', { detail: { text: t } })));
    } catch (err) {
      showFeedback(`Failed to clear conversation: ${err.message}`, 'error');
    }
  });
}

// STEP 5: Load document handler
async function handleLoadDocument() {
  const input = document.getElementById('doc-url-input');
  const url = (input?.value || '').trim();

  if (!url) {
    showFeedback('Please enter a Google Doc URL', 'error');
    return;
  }

  if (!isValidGoogleDocUrl(url)) {
    showFeedback('Invalid Google Doc URL. Please use a publicly accessible document link.', 'error');
    return;
  }

  debugLog('Loading document from URL:', url);
  showLoading();

  try {
    const resp = await ingestDocument(url);
    const count = resp?.chunks_created ?? resp?.total_chunks ?? 0;

    // Switch to chat UI after successful ingestion
    switchToChat();
    updateStatus(true);

    // Show stats and success message
    showDocStats(count);

    // Greeting
    appendMessage({
      role: 'assistant',
      text: `🎉 Success! I've analyzed your document into ${count} sections.\n\nI can now:\n✓ Answer questions about any part\n✓ Find specific information instantly\n✓ Cite exact sources for every answer\n\nTry asking:`
    });

    // Show example chips
    showExampleQuestions([
      "What is the main topic?",
      "Summarize the key points",
      "What are the conclusions?"
    ], (t) => window.dispatchEvent(new CustomEvent('rag:sendMessage', { detail: { text: t } })));

  } catch (err) {
    showFeedback(`Failed to load document: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

// STEP 6: Startup code
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

// STEP 7: Export debug info
window.ragChatbot = {
  config: CONFIG,
  sessionId,
  version: 'v1.0'
};

// STEP 8: Log module load
console.log('Main module loaded');