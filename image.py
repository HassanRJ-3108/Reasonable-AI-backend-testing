from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import uuid
from PIL import Image
import io
from dotenv import load_dotenv

from database import get_db
from models import User
from schemas import ImageUploadResponse
from auth import get_current_user

# Import the image processor
from ai.image_processor import ImageProcessor

# Load environment variables
load_dotenv()

# Router
router = APIRouter(
    prefix="/api/image",
    tags=["image"],
    responses={401: {"description": "Unauthorized"}},
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize image processor if API key is available
image_processor = None
if GEMINI_API_KEY:
    image_processor = ImageProcessor(GEMINI_API_KEY, model_name="gemini-2.0-flash")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join("uploads", unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Return file URL
    # In a production environment, you would use a proper file storage service
    # and return a public URL. For this example, we'll just return the local path.
    return {"image_url": file_path}

@router.post("/analyze", response_model=dict)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    # Check if Gemini API key is configured
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API key not configured"
        )
    
    # Initialize image processor if not already initialized
    global image_processor
    if image_processor is None:
        image_processor = ImageProcessor(GEMINI_API_KEY, model_name="gemini-2.0-flash")
    
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Read image
        image_content = await file.read()
        
        # Process image using the image processor
        config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192
        }
        
        analysis = image_processor.analyze_image(image_content, prompt, config)
        
        return {"analysis": analysis}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing image: {str(e)}"
        )

