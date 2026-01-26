import { API_BASE_URL } from './config.js';

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
  chat: (message, session_id = 'default') => request('/chat', { method: 'POST', body: JSON.stringify({ message, session_id }) }),
  clear: (session_id = 'default') => request('/clear', { method: 'POST', body: JSON.stringify({ session_id }) }),
};