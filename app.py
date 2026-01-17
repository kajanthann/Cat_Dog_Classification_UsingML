from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained model
model = load_model("./model/best_mobilenet_model.h5")

TARGET_SIZE = (224,224)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Load & preprocess image
    img = image.load_img(file_path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    pred = model.predict(x)[0][0]

    # Optional thresholds
    cat_threshold = 0.1
    dog_threshold = 0.95

    if pred < cat_threshold:
        result = "Cat 🐱"
    elif pred > dog_threshold:
        result = "Dog 🐶"
    else:
        result = f"This is not a cat or dog ({pred:.2f})"

    return templates.TemplateResponse("index.html", {
    "request": request,
    "result": result,
    "image_path": "/" + file_path.replace("\\", "/"),
    "probability": float(pred)
    })
