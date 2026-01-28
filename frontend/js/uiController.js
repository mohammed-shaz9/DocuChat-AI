export class UIController {
  constructor() {
    this.messagesEl = document.getElementById('chat-messages');
    this.inputEl = document.getElementById('user-input');
    this.sendBtn = document.getElementById('send-btn');
  }
  appendMessage({ role, text }) {
    if (!this.messagesEl) return; // Guard against missing DOM element
    const el = document.createElement('div');
    el.className = `message ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    el.appendChild(bubble);
    this.messagesEl.appendChild(el);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }
  onSubmit(handler) {
    // Click on send button
    if (this.sendBtn) {
      this.sendBtn.addEventListener('click', async () => {
        const value = (this.inputEl?.value || '').trim();
        if (!value) return;
        await handler(value);
        if (this.inputEl) this.inputEl.value = '';
      });
    }
    // Enter key in textarea
    if (this.inputEl) {
      this.inputEl.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          const value = (this.inputEl.value || '').trim();
          if (!value) return;
          await handler(value);
          this.inputEl.value = '';
        }
      });
    }
  }
}

// NEW: Functional UI helpers for main.js
let uiRefs = null;
export function initializeUI() {
  uiRefs = {
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    ingestFeedback: document.getElementById('ingest-feedback'),
    chatMessages: document.getElementById('chat-messages'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    clearBtn: document.getElementById('clear-chat-btn'),
    loadDocBtn: document.getElementById('load-doc-btn'),
    docUrlInput: document.getElementById('doc-url-input'),
    ingestSection: document.getElementById('ingest-section'),
    chatSection: document.getElementById('chat-section'),
    loadingOverlay: document.getElementById('loading-overlay'),
    toastContainer: document.getElementById('toast-container'),
  };
}

export function showLoading() {
  if (!uiRefs) initializeUI();
  if (uiRefs.loadingOverlay) uiRefs.loadingOverlay.style.display = 'flex';
}
export function hideLoading() {
  if (!uiRefs) initializeUI();
  if (uiRefs.loadingOverlay) uiRefs.loadingOverlay.style.display = 'none';
}

export function updateStatus(connected, info) {
  if (!uiRefs) initializeUI();
  if (uiRefs.statusDot) uiRefs.statusDot.style.backgroundColor = connected ? '#10b981' : '#ef4444';
  if (uiRefs.statusText) uiRefs.statusText.textContent = connected ? 'Connected' : 'Not Connected';
}


export function showExampleQuestions(questions, handler) {
  if (!uiRefs) initializeUI();
  const container = uiRefs.chatMessages;
  if (!container) return;

  const wrapper = document.createElement('div');
  wrapper.className = 'suggestions-wrapper';
  wrapper.style.marginBottom = '20px';
  wrapper.style.display = 'flex';
  wrapper.style.gap = '8px';
  wrapper.style.flexWrap = 'wrap';

  questions.forEach(q => {
    const btn = document.createElement('button');
    btn.className = 'btn btn-secondary btn-small suggestion-chip';
    btn.textContent = q;
    btn.style.borderRadius = '20px';
    btn.style.fontSize = '0.85rem';
    btn.onclick = () => handler(q);
    wrapper.appendChild(btn);
  });

  // Append to chat (or specific area if we had one)
  container.appendChild(wrapper);
  container.scrollTop = container.scrollHeight;
}

export function showDocStats(count) {
  if (!uiRefs) initializeUI();
  // We can insert this into the chat or replace the ingest section content
  // For now, let's show a toast and maybe a persistent indicator in chat
  showFeedback(`Verified ${count} sections. Ready to answer! 🚀`, 'success');
}

export function showFeedback(message, type = 'info') {
  if (!uiRefs) initializeUI();

  // Add emojis based on type
  let emoji = 'ℹ️';
  if (type === 'success') emoji = '✅';
  if (type === 'error') emoji = '❌';
  if (type === 'warning') emoji = '⚠️';

  showToast(`${emoji} ${message}`, type);

  const el = uiRefs.ingestFeedback;
  if (el) {
    el.textContent = `${emoji} ${message}`;
    el.className = `feedback show ${type}`;
  }
}

export function showToast(message, type = 'info') {
  if (!uiRefs) initializeUI();
  const container = uiRefs.toastContainer;
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateY(-6px)'; }, 3500);
  setTimeout(() => { toast.remove(); }, 4000);
}

export function switchToChat() {
  if (!uiRefs) initializeUI();
  if (uiRefs.ingestSection) uiRefs.ingestSection.style.display = 'none';
  if (uiRefs.chatSection) uiRefs.chatSection.style.display = 'block';
}

export function appendMessage({ role, text, confidence, sources }) {
  if (!uiRefs) initializeUI();
  const container = uiRefs.chatMessages;
  if (!container) return;

  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  // Render text with simple formatting (could use markdown lib later)
  bubble.textContent = text;

  // Add citations/confidence if available (Assistant only)
  if (role === 'assistant') {
    // Confidence
    if (confidence) {
      const confBadge = document.createElement('div');
      confBadge.className = `confidence-badge ${confidence}`;
      // style handled in CSS or inline for now
      confBadge.style.fontSize = '0.75rem';
      confBadge.style.marginTop = '4px';
      confBadge.style.opacity = '0.8';
      if (confidence === 'high') confBadge.innerHTML = '✓ High Confidence';
      else if (confidence === 'medium') confBadge.innerHTML = 'ℹ️ Medium Confidence';
      else if (confidence === 'clarify') confBadge.innerHTML = '🤔 Needs Clarification';
      else confBadge.innerHTML = '🔍 Low Confidence';
      bubble.appendChild(confBadge);
    }
  }

  wrap.appendChild(bubble); // Bubble contains text + metadata
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
}