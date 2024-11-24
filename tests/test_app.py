import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint_no_file():
    response = client.post("/predict")
    assert response.status_code == 422  
    assert "detail" in response.json()

def test_predict_endpoint_with_invalid_file():
    invalid_file = {"file": ("test.txt", b"This is not an image.", "text/plain")}
    response = client.post("/predict", files=invalid_file)
    assert response.status_code == 400  
    assert "detail" in response.json()

def test_predict_endpoint_with_valid_image():
    from PIL import Image
    import io

    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    image_bytes = io.BytesIO()
    dummy_image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    valid_file = {"file": ("test.png", image_bytes, "image/png")}
    response = client.post("/predict", files=valid_file)

    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert "superimposed_img" in response_json
    assert response_json["prediction"] in ["High Crack", "Low Crack", "Medium Crack", "No Crack"]

def test_healthcheck():
    response = client.get("/")
    assert response.status_code == 404  
