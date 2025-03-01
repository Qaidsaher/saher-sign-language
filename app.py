from fastapi import FastAPI, UploadFile, File, HTTPException
from inference_sdk import InferenceHTTPClient
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the InferenceHTTPClient with your API details.
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="uLtdkU4nhEDy26911Ycc"
)

MODEL_ID = "arabic-sign-language-translator/2"

@app.get("/")
async def root():
    return {"message": "ONNX Model API is running"}

@app.get("/saher-test")
async def saher_test():
    return "this is saher test sign language API running test"

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image file upload, saves it temporarily,
    and uses the inference-sdk to run predictions on the image.
    """
    try:
        # Save the uploaded file to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_filename = tmp.name

        # Run inference using the inference-sdk.
        result = CLIENT.infer(temp_filename, model_id=MODEL_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        # Clean up the temporary file.
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    return result
