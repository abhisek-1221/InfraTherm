const API_URL = 'http://localhost:8000';

export async function predictImage(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to predict image');
  }

  return response.json();
}

