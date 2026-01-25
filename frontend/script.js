const messagesEl = document.getElementById('messages');
const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');

function addMessage(text, role){
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if(!text) return;
  addMessage(text, 'user');
  input.value = '';
  // Phase 1: no backend call yet. Placeholder bot response.
  setTimeout(() => addMessage('Backend not wired yet — skeleton ready.', 'bot'), 300);
});