
---

# Infratherm 

Infratherm is an **AI-powered application** implementing the research paper:
## 📖 Research Paper Citation  
> [Infrared Thermal Imaging for Pavement Inspection with Hierarchical Hybrid Vision Transformers](https://drive.google.com/file/d/19WPRX4OB4TQgh2yZEe_iIgsCUdZRDYbc/view)

![image](https://github.com/user-attachments/assets/0a33313f-6996-431d-bfda-2e426d30ce91)


> It benchmarks **CNN, ViT, and hybrid ViT-CNN architectures** for detecting pavement cracks in **Infrared Thermal (IRT) images**, aiding in efficient infrastructure maintenance.  

## ✨ Features  

### 🔍 AI-Powered Pavement Crack Detection  
- Supports multiple **Deep Learning models**:  
  - **CNNs**: ResNet50, InceptionV3, EfficientNet-B0, VGG19  
  - **ViTs**: ViT-B/16  
  - **Hybrid ViT-CNN Architectures**: CoAtNet-3, MaxViT  

### 🔬 Model Interpretability  
- Utilizes **Grad-CAM** for **visualizing model decisions** and crack detection areas.
- ![image](https://github.com/user-attachments/assets/c81586b9-45b7-43cd-b8dd-ff80f3debd6a)
  

### ⚡ Efficient Infrastructure Maintenance  
- Benchmarks **model performance** for selecting the best architecture.
  ![image](https://github.com/user-attachments/assets/50fbdf9b-8282-4dd5-bdcb-ff543c820b03)
  
 ## 🛠 Tech Stack  
- **Frontend**: Next.js  
- **Backend**: FastAPI  
- OpenCV  
- TensorFlow  
- Docker  
- TypeScript  
- Python  

## 🚀 Get Started  

### 1️⃣ Clone the repository  
```sh
git clone https://github.com/abhisek-1221/Infratherm.git
cd Infratherm
```

### 2️⃣ Setup & Run Backend (FastAPI)  

#### ➤ Create a Virtual Environment  

For **macOS/Linux**:  
```sh
cd server
python3 -m venv venv
source venv/bin/activate
```

For **Windows (CMD/PowerShell)**:  
```sh
cd server
python -m venv venv
venv\Scripts\activate
```

#### ➤ Install dependencies  
```sh
pip install -r requirements.txt
```

#### ➤ Run FastAPI Server  
```sh
uvicorn app:main --host 0.0.0.0 --port 8000 --reload
```

---

### 3️⃣ Setup & Run Frontend (Next.js)  
```sh
cd ../client
npm install
npm run dev
```

### 4️⃣ Access the App  
```
http://localhost:3000
```

---

## 🤝 Contribute  
Infratherm is **open-source**! Contributions to improve model performance, interpretability, and UI/UX are welcome.  

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Add new feature"`)  
4. Push to branch (`git push origin feature-name`)  
5. Open a Pull Request  


## ⭐ Star the Repo  
If you find **Infratherm** useful, don’t forget to **star ⭐ the repository** on GitHub!  

---
