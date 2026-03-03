const API_BASE = '/api';

export async function analyzeAudioAPI(audioFile) {
  const formData = new FormData();
  formData.append('file', audioFile);

  const res = await fetch(`${API_BASE}/analyze/audio`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(err.detail || `Error ${res.status}`);
  }

  return res.json();
}

export async function analyzeTextAPI(text, language = 'en') {
  const res = await fetch(`${API_BASE}/analyze/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(err.detail || `Error ${res.status}`);
  }

  return res.json();
}