from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

# Import routers
from auth import router as auth_router
from chat import router as chat_router
from image import router as image_router
from config import router as config_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Reasonable AI API",
    description="Backend API for Reasonable AI, an autonomous reasoning chatbot created by Hassan RJ - Full Stack Nextjs Developer | Python developer | Student Leader | Web 3.0 | Generative AI | Agentic AI | AI Assistants | AI Powered Chatbots | Building Dynamic Web Apps | Learning & Growing!",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Reasonable AI API",
        "version": "1.0.0",
        "documentation": "/docs",
        "creator": "Hassan RJ - Full Stack Nextjs Developer | Python developer | Student Leader | Web 3.0 | Generative AI | Agentic AI | AI Assistants | AI Powered Chatbots | Building Dynamic Web Apps | Learning & Growing!"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(image_router)
app.include_router(config_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get port from Railway, default to 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port)