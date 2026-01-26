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