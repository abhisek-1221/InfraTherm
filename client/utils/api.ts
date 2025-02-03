const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/predict';

export async function predictImage(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to predict image');
  }

  return response.json();
}

