import { api } from './api.js';
import { UIController } from './uiController.js';
import { ChatManager } from './chatManager.js';

const ui = new UIController();
const chat = new ChatManager();

ui.onSubmit(async (text) => {
  ui.appendMessage(chat.addUserMessage(text));
  try {
    const res = await api.chat(text);
    const answer = res?.answer || '[No answer]';
    ui.appendMessage(chat.addAssistantMessage(answer));
  } catch (err) {
    ui.appendMessage(chat.addAssistantMessage(`Error: ${err.message}`));
  }
});

// Optionally show health on load
(async () => {
  try {
    const h = await api.health();
    ui.appendMessage({ role: 'assistant', text: `Server: ${h.status}, chunks: ${h.total_chunks}` });
  } catch (err) {
    ui.appendMessage({ role: 'assistant', text: `Health check failed: ${err.message}` });
  }
})();