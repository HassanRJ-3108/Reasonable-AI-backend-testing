from pydantic import BaseModel, EmailStr, Field, UUID4
from typing import List, Optional, Union
from datetime import datetime

# Auth schemas
class UserBase(BaseModel):
    username: str
    email: str  # Changed from EmailStr to str for simplicity

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: str  # Changed from UUID4 to str for simplicity
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Configuration schemas
class ConfigurationBase(BaseModel):
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    enable_reasoning: bool = True
    max_iterations: int = 3
    top_p: float = 0.95
    top_k: int = 64

class ConfigurationUpdate(ConfigurationBase):
    pass

class ConfigurationResponse(ConfigurationBase):
    user_id: str  # Changed from UUID4 to str for simplicity

    class Config:
        orm_mode = True

# Model info schema
class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    max_tokens: int
    supports_images: bool

# Chat schemas
class MessageCreate(BaseModel):
    content: str
    image_data: Optional[str] = None  # Base64 encoded image data

# Make sure ReasoningStepResponse is defined before it's used
class ReasoningStepResponse(BaseModel):
    id: str  # Changed from UUID4 to str for simplicity
    step_number: int
    content: str
    type: str

    class Config:
        orm_mode = True

class MessageResponse(BaseModel):
    id: str  # Changed from UUID4 to str for simplicity
    role: str
    content: str
    timestamp: datetime
    image_url: Optional[str] = None
    reasoning_steps: Optional[List[ReasoningStepResponse]] = None

    class Config:
        orm_mode = True

class ConversationBase(BaseModel):
    title: str

class ConversationCreate(ConversationBase):
    pass

class ConversationResponse(ConversationBase):
    id: str  # Changed from UUID4 to str for simplicity
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        orm_mode = True

class ConversationListResponse(BaseModel):
    id: str  # Changed from UUID4 to str for simplicity
    title: str
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None

    class Config:
        orm_mode = True

# Streaming response schemas
class ThinkingStepUpdate(BaseModel):
    message_id: str
    step_number: int
    content: str
    type: str
    is_complete: bool = False

# Image schemas
class ImageUploadResponse(BaseModel):
    image_url: str

