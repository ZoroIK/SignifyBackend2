import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ Define ASL Alphabet Mapping
ASL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]

# 🎯 FastAPI App
app = FastAPI()

# ✅ CORS Middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for security in production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ✅ Load the TensorFlow model (now local)
MODEL_PATH = "asl_cnn_2D_model.h5"

if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found. Please make sure 'asl_cnn_2D_model.h5' is in the project directory.")
    exit(1)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ 2D Model Loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# 📦 Input schema
class LandmarkData(BaseModel):
    landmarks: list  # 42 values expected

# 🧠 Prediction Endpoint
@app.post("/predict")
async def predict(data: LandmarkData):
    if len(data.landmarks) != 42:
        raise HTTPException(status_code=400, detail="Expected 42 landmark values (x, y for 21 points).")

    try:
        input_data = np.array([data.landmarks])
        predictions = model.predict(input_data, verbose=0)[0]
        label_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        if confidence < 0.2:
            return {"prediction": "Unknown", "confidence": confidence}

        predicted_letter = ASL_LETTERS[label_index] if label_index < len(ASL_LETTERS) else "?"

        return {
            "prediction": predicted_letter,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
