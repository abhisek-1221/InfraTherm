import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

model = load_model('./models/gradcam.h5')
last_conv_layer_name = "top_conv"
class_names = ["High Crack", "Low Crack", "Medium Crack", "No Crack"]

def grad_cam(model, img_array, class_idx, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def predict_single_image(image, model, last_conv_layer_name, class_name):
    img_array = np.expand_dims(img_to_array(image) / 255.0, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    heatmap = grad_cam(model.get_layer("efficientnetb0"), img_array, predicted_class, last_conv_layer_name)
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    alpha = 0.8
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, alpha, 0)

    return predictions, predicted_class, superimposed_img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
    
    predictions, predicted_class, superimposed_img = predict_single_image(image, model, last_conv_layer_name, class_names)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    superimposed_img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "prediction": class_names[predicted_class],
        "superimposed_img": superimposed_img_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)