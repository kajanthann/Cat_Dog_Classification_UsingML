# 🐶 Cat vs Dog Classification System 🐱

A deep learning-based image classification system that predicts whether an uploaded image contains a **cat** or a **dog** using a **MobileNetV2** Convolutional Neural Network (CNN). The project includes model development, training, and a user-friendly web interface built with **FastAPI** and **Tailwind CSS**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This project implements a **binary image classification system** using transfer learning with **MobileNetV2**, a lightweight yet powerful pre-trained deep neural network. The system can classify images as either containing a **cat** (class 0) or a **dog** (class 1) with high accuracy.

### Key Objectives:
- Build a robust binary classifier for cats and dogs
- Leverage transfer learning to improve training efficiency
- Deploy the model via a web interface for easy accessibility
- Containerize the application using Docker for seamless deployment

---

## ✨ Key Features

✅ **Transfer Learning**: Uses pre-trained MobileNetV2 weights from ImageNet  
✅ **Fine-tuning**: Last 50 layers unfrozen for domain-specific optimization  
✅ **Data Augmentation**: Rotation, shift, zoom, and flip for better generalization  
✅ **Web Interface**: Modern, interactive UI with real-time predictions  
✅ **FastAPI Backend**: RESTful API with automatic OpenAPI documentation  
✅ **Docker Support**: Full containerization for production deployment  
✅ **Early Stopping**: Prevents overfitting with patience-based monitoring  
✅ **Model Checkpointing**: Saves the best model during training  
✅ **Learning Rate Reduction**: Adaptive learning rate scheduling  

---

## 🏗️ Technical Architecture

### System Components:

```
┌─────────────────────────────────────────┐
│         Web Interface (HTML/CSS)        │
│      Tailwind CSS with Responsive UI    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    FastAPI Backend (Python)             │
│  - Image Upload & Preprocessing         │
│  - Model Inference & Prediction         │
│  - Response Rendering                   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    TensorFlow/Keras Model               │
│    - MobileNetV2 + Custom Layers        │
│    - Binary Classification (Cat/Dog)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
        Model: best_mobilenet_model.h5
```

---

## 📊 Dataset Information

The project uses the **Dog and Cat Classification Dataset** from Kaggle, organized into **train**, **validation**, and **test** splits.

### Dataset Statistics:

| Split | Cats (Class 0) | Dogs (Class 1) | Total |
|-------|---|---|---|
| **Train** | ~2500 | ~2500 | ~5000 |
| **Validation** | ~600 | ~600 | ~1200 |
| **Test** | ~600 | ~600 | ~1200 |
| **Total** | ~3700 | ~3700 | ~7400 |

### Data Preprocessing:

1. **Image Resizing**: All images resized to 224×224 pixels (MobileNetV2 input requirement)
2. **Color Conversion**: Images converted to RGB format if needed
3. **Normalization**: Pixel values scaled to [0, 1] range

### Data Augmentation (Training Only):

- **Rotation**: 20° random rotation
- **Width/Height Shift**: 20% random shifts
- **Shear**: 20% shear transformation
- **Zoom**: 20% random zoom
- **Horizontal Flip**: Random horizontal flips
- **Fill Mode**: Nearest pixel fill for new areas

---

## 🧠 Model Architecture

### Base Model: **MobileNetV2**

MobileNetV2 is a lightweight, efficient architecture designed for mobile and edge devices:
- **Input Shape**: (224, 224, 3)
- **Pre-trained Weights**: ImageNet
- **Depthwise Separable Convolutions**: Reduced computation and memory
- **Inverted Residual Blocks**: Efficient feature extraction

### Custom Classification Head:

```
MobileNetV2 (pre-trained, frozen initially)
        ↓
Global Average Pooling 2D
        ↓
Batch Normalization
        ↓
Dense Layer (256 units, ReLU activation)
        ↓
Dropout (0.3)
        ↓
Output Layer (1 unit, Sigmoid activation) → Binary Classification
```

### Training Strategy:

**Phase 1: Transfer Learning (15 epochs)**
- Freeze all MobileNetV2 layers
- Train only the custom classification head
- Learning Rate: 0.0001 (Adam optimizer)
- Objective: Adapt pre-trained features to cat/dog domain

**Phase 2: Fine-tuning (10 epochs)**
- Unfreeze last 50 layers of MobileNetV2
- Train entire model with lower learning rate
- Learning Rate: 1e-5 (Adam optimizer)
- Objective: Fine-tune features for better accuracy

### Loss Function:
**Binary Crossentropy** - Standard for binary classification

### Metrics:
**Accuracy** - Primary evaluation metric

---

## ⚙️ Installation & Setup

### Prerequisites:

- Python 3.8+
- pip (Python package manager)
- Docker (optional, for containerized deployment)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dogvscat.git
cd dogvscat
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages:

```
fastapi                 # Web framework
uvicorn[standard]       # ASGI server
jinja2                  # Template rendering
tensorflow==2.13.0      # Deep learning framework
numpy                   # Numerical computations
pillow                  # Image processing
```

### Step 3: Verify Model Files

Ensure `best_mobilenet_model.h5` exists in the `model/` directory:

```bash
ls -la model/
```

Expected files:
- `best_mobilenet_model.h5` (Primary trained model)
- `mobilenetv2_cat_dog.keras` (Alternative format)
- `model.h5` (Backup model)
- `model.keras` (Backup format)

---

## 🚀 Usage

### Running the Application Locally

#### Using Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Using Python:

```bash
python -m uvicorn app:app --reload
```

### Accessing the Web Interface

Open your browser and navigate to:

```
http://localhost:8000
```

### Making Predictions

1. **Upload an Image**: Click the upload box to select a cat or dog image
2. **Preview**: Image preview displays immediately
3. **Click Predict**: Submit the image for classification
4. **View Results**: Get instant prediction with confidence score
5. **Reset**: Click Reset to classify another image

### API Endpoints

#### GET `/`
- **Description**: Render the main classification interface
- **Response**: HTML page with upload form

#### POST `/predict/`
- **Description**: Process uploaded image and return prediction
- **Request**: 
  - `file`: Image file (multipart/form-data)
- **Response**: HTML page with prediction result and probability

#### Example cURL Request:

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@cat.jpg"
```

---

## 📁 Project Structure

```
dogvscat/
├── app.py                          # FastAPI application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
│
├── model/                          # Trained models
│   ├── best_mobilenet_model.h5     # Primary model
│   ├── mobilenetv2_cat_dog.keras   # Alternative format
│   ├── model.h5                    # Backup model
│   └── model.keras                 # Backup format
│
├── notebook/                       # Jupyter notebooks
│   ├── cat_dog_model.ipynb        # Model development & training
│   └── model.ipynb                # Additional experiments
│
├── templates/                      # HTML templates
│   └── index.html                 # Main UI
│
├── static/                         # Static assets
│   ├── index.css                  # Styling
│   ├── uploads/                   # Uploaded images (auto-created)
│   └── bg.png                     # Background image
│
├── data/                           # Dataset
│   ├── train/                     # Training images (0: cat, 1: dog)
│   │   ├── 0/
│   │   └── 1/
│   ├── val/                       # Validation images
│   │   ├── 0/
│   │   └── 1/
│   ├── test/                      # Test images
│   │   ├── 0/
│   │   └── 1/
│   └── resized_imgs/              # Preprocessed images
│       ├── train/
│       ├── val/
│       └── test/
│
└── test/                           # Test dataset (standalone)
    ├── cats/
    └── dogs/
```

---

## 📈 Model Performance

### Evaluation Metrics:

The model is evaluated on the **test set** using the following metrics:

- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Breakdown of correct and incorrect predictions
- **ROC Curve**: Trade-off between true positive and false positive rates
- **AUC (Area Under Curve)**: Measure of overall model performance

### Prediction Thresholds:

The model uses adjustable confidence thresholds for classification:

```python
cat_threshold = 0.1    # Below this: Predicted as Cat
dog_threshold = 0.95   # Above this: Predicted as Dog
Between thresholds: "Not a cat or dog"
```

### Expected Performance:

- **Accuracy**: ~92-96%
- **Precision**: ~91-95%
- **Recall**: ~90-94%
- **F1-Score**: ~91-94%

---

## 🐳 Docker Deployment

### Building Docker Image

```bash
docker build -t dogvscat:latest .
```

### Running Container

```bash
docker run -p 8000:8000 dogvscat:latest
```

### Docker Configuration:

- **Base Image**: Python 3.13.9-slim
- **Exposed Port**: 8000
- **Working Directory**: /app
- **Entrypoint**: `uvicorn app:app --host 0.0.0.0 --port 8000`
- **Upload Directory**: `/app/static/uploads` (auto-created)

### Dockerfile Highlights:

- Lightweight base image for smaller container size
- System dependencies installed (build-essential)
- PIP cache cleared to reduce image size
- Upload directory created automatically
- Production-ready ASGI server (Uvicorn)

---

## 🛠️ Technologies Used

### Deep Learning:
- **TensorFlow/Keras**: Model development and training
- **MobileNetV2**: Pre-trained transfer learning architecture
- **NumPy**: Numerical computations

### Backend:
- **FastAPI**: Modern, fast web framework
- **Uvicorn**: ASGI web server
- **Jinja2**: Template rendering

### Frontend:
- **HTML5**: Page structure
- **CSS (Tailwind)**: Responsive styling
- **JavaScript**: Client-side interactions

### Image Processing:
- **Pillow (PIL)**: Image manipulation

### Containerization:
- **Docker**: Application containerization

### Development:
- **Jupyter Notebook**: Model experimentation
- **Kaggle API**: Dataset download

---

## 🔧 API Response Examples

### Successful Prediction - Cat:

```json
{
  "result": "Cat 🐱",
  "image_path": "/static/uploads/cat_image.jpg",
  "probability": 0.08
}
```

### Successful Prediction - Dog:

```json
{
  "result": "Dog 🐶",
  "image_path": "/static/uploads/dog_image.jpg",
  "probability": 0.97
}
```

### Uncertain Prediction:

```json
{
  "result": "This is not a cat or dog (0.45)",
  "image_path": "/static/uploads/unknown_image.jpg",
  "probability": 0.45
}
```

---

## 🚀 Future Improvements

### Model Enhancements:
- [ ] Implement multi-class classification (add more animal types)
- [ ] Use EfficientNet or Vision Transformer for better accuracy
- [ ] Add confidence scoring and uncertainty quantification
- [ ] Implement ensemble methods for improved robustness

### Features:
- [ ] Add batch prediction for multiple images
- [ ] Implement image preprocessing options (crop, rotate, etc.)
- [ ] Add model explainability (Grad-CAM visualization)
- [ ] Support for video input and frame-by-frame classification

### Deployment:
- [ ] Add authentication and rate limiting
- [ ] Implement logging and monitoring
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Create mobile app version
- [ ] Add caching for frequent predictions

### Code Quality:
- [ ] Add comprehensive unit and integration tests
- [ ] Implement CI/CD pipeline
- [ ] Add API documentation (Swagger/OpenAPI)
- [ ] Code optimization and profiling

---

## 📝 Training Details

### Data Sources:
- **Dataset**: Dog and Cat Classification Dataset (Kaggle)
- **Download**: Using Kaggle API

### Training Environment:
- **Framework**: TensorFlow 2.13.0
- **GPU Support**: CUDA-enabled (if available)
- **Callbacks Used**:
  - **EarlyStopping**: Stops training when validation loss plateaus (patience: 5 epochs)
  - **ModelCheckpoint**: Saves best model based on validation loss
  - **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

### Hyperparameters:

| Parameter | Value |
|---|---|
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Learning Rate (Phase 1) | 0.0001 |
| Learning Rate (Phase 2) | 1e-5 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs (Phase 1) | 15 |
| Epochs (Phase 2) | 10 |
| Dropout Rate | 0.3 |
| Dense Units | 256 |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with improvements or bug fixes.

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the project maintainer

---

## 🙏 Acknowledgments

- **Kaggle**: For providing the dog and cat classification dataset
- **TensorFlow/Keras**: For the deep learning framework
- **MobileNet Team**: For the efficient MobileNetV2 architecture
- **FastAPI Community**: For the amazing web framework

---

**Last Updated**: January 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
