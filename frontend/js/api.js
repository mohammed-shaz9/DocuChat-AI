import { API_BASE_URL, getSessionId } from './config.js';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export const api = {
  health: () => request('/health'),
  ingest: (doc_url) => request('/ingest', { method: 'POST', body: JSON.stringify({ doc_url }) }),
  chat: (message, session_id = getSessionId()) => request('/chat', { method: 'POST', body: JSON.stringify({ message, session_id }) }),
  clear: (session_id = getSessionId()) => request('/clear', { method: 'POST', body: JSON.stringify({ session_id }) }),
};

// NEW: Named exports used by main.js
export function checkHealth() {
  return api.health();
}
export function ingestDocument(doc_url) {
  return api.ingest(doc_url);
}
export function sendChat(message, session_id = getSessionId()) {
  return api.chat(message, session_id);
}
export function clearConversation(session_id = getSessionId()) {
  return api.clear(session_id);
}