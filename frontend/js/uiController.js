export class UIController {
  constructor() {
    this.messagesEl = document.getElementById('messages');
    this.formEl = document.getElementById('chat-form');
    this.inputEl = document.getElementById('user-input');
  }
  appendMessage({ role, text }) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    el.textContent = text;
    this.messagesEl.appendChild(el);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }
  onSubmit(handler) {
    this.formEl.addEventListener('submit', async (e) => {
      e.preventDefault();
      const value = this.inputEl.value.trim();
      if (!value) return;
      await handler(value);
      this.inputEl.value = '';
    });
  }
}