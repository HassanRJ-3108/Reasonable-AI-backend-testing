from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from database import get_db
from models import Configuration, User
from schemas import ConfigurationResponse, ConfigurationUpdate, ModelInfo
from auth import get_current_user

# Router
router = APIRouter(
    prefix="/api/config",
    tags=["configuration"],
    responses={401: {"description": "Unauthorized"}},
)

# Available models
AVAILABLE_MODELS = [
    ModelInfo(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        description="Fast model with good performance",
        max_tokens=8192,
        supports_images=True
    ),
    ModelInfo(
        id="gemini-2.0-pro",
        name="Gemini 2.0 Pro",
        description="Balanced model with excellent quality",
        max_tokens=8192,
        supports_images=True
    ),
    ModelInfo(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        description="Legacy model",
        max_tokens=8192,
        supports_images=True
    ),
]

# Helper functions
def get_user_config(db: Session, user_id):
    return db.query(Configuration).filter(Configuration.user_id == user_id).first()

def create_default_config(db: Session, user_id):
    config = Configuration(user_id=user_id)
    db.add(config)
    db.commit()
    db.refresh(config)
    return config

# Endpoints
@router.get("", response_model=ConfigurationResponse)
async def get_configuration(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    config = get_user_config(db, current_user.id)
    if not config:
        config = create_default_config(db, current_user.id)
    return config

@router.put("", response_model=ConfigurationResponse)
async def update_configuration(
    config_update: ConfigurationUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    config = get_user_config(db, current_user.id)
    if not config:
        config = create_default_config(db, current_user.id)
    
    # Update configuration
    for key, value in config_update.dict().items():
        setattr(config, key, value)
    
    db.commit()
    db.refresh(config)
    return config

@router.get("/models", response_model=List[ModelInfo])
async def get_available_models(
    current_user: User = Depends(get_current_user)
):
    return AVAILABLE_MODELS

