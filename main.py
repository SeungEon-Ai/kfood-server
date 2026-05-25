from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import io

app = FastAPI(title="한식 분류 API")

# CORS 설정 (모바일 앱에서 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 + 클래스 이름 로드 (서버 시작 시 1번만)
print("Loading model...")
model = tf.keras.models.load_model('best.keras')
with open('class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
print(f"Model loaded. {len(class_names)} classes ready.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "classes": len(class_names),
        "message": "한식 분류 API 작동 중"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """이미지 받아서 Top-5 음식 예측 반환"""
    try:
        # 1. 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 2. 전처리 (학습 시와 동일)
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        # Rescaling: [0, 255] -> [-1, 1]
        # img_array = img_array / 127.5 - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. 추론
        predictions = model.predict(img_array, verbose=0)[0]
        
        # 4. Top-5 추출
        top5_idx = np.argsort(predictions)[-5:][::-1]
        results = [
            {
                "food": class_names[int(i)],
                "confidence": float(predictions[i])
            }
            for i in top5_idx
        ]
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))