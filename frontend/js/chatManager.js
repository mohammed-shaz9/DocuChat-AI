export class ChatManager {
  constructor() {
    this.messages = [];
  }
  addUserMessage(text) {
    this.messages.push({ role: 'user', text });
    return { role: 'user', text };
  }
  addAssistantMessage(text) {
    this.messages.push({ role: 'assistant', text });
    return { role: 'assistant', text };
  }
  getHistory() {
    return this.messages.slice();
  }
}

// NEW: procedural handlers for main.js compatibility
export function handleSendMessage() {
  const input = document.getElementById('user-input');
  const value = (input?.value || '').trim();
  if (!value) return;
  const evt = new CustomEvent('rag:sendMessage', { detail: { text: value } });
  window.dispatchEvent(evt);
}

export function handleClearChat() {
  const evt = new CustomEvent('rag:clearChat');
  window.dispatchEvent(evt);
}