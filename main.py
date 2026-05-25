from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
import numpy as np
import json
import io
import os

app = FastAPI(title="한식 분류 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TFLite 모델 로드
print("Loading TFLite model...")
interpreter = Interpreter(model_path='kfood_dynamic.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 클래스 이름 로드
with open('class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
print(f"Model loaded. {len(class_names)} classes ready.")

# 영양정보 로드
with open('nutrition_data.json', 'r', encoding='utf-8') as f:
    nutrition_data = json.load(f)
print(f"Nutrition data loaded. {len(nutrition_data)} foods.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "classes": len(class_names),
        "nutrition": len(nutrition_data),
        "message": "한식 분류 API 작동 중"
    }


@app.get("/nutrition/{food_name}")
def get_nutrition(food_name: str):
    """음식 이름으로 영양정보 조회"""
    if food_name in nutrition_data:
        return {
            "name_ko": food_name,
            **nutrition_data[food_name]
        }
    raise HTTPException(status_code=404, detail=f"'{food_name}' 영양정보 없음")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """이미지 받아서 Top-5 음식 예측 반환"""
    try:
        # 1. 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 2. 전처리 (정규화 빼기 - 모델 내부에 Rescaling 있음)
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        # img_array = img_array / 127.5 - 1.0  # 주석 처리
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. TFLite 추론
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)