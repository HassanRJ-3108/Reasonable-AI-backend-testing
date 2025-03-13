from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import uuid
import json
import base64
import asyncio
from datetime import datetime
import os
import io
from PIL import Image
from dotenv import load_dotenv

from database import get_db
from models import User, Conversation, Message, ReasoningStep, Configuration
from schemas import (
    ConversationCreate, 
    ConversationResponse, 
    ConversationListResponse,
    MessageCreate, 
    MessageResponse,
    ReasoningStepResponse,
    ThinkingStepUpdate
)
from auth import get_current_user

# Import the reasoning system and image processor
from ai.reasoning import ReasoningSystem
from ai.image_processor import ImageProcessor

# Load environment variables
load_dotenv()

# Router
router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    responses={401: {"description": "Unauthorized"}},
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize reasoning system and image processor if API key is available
reasoning_system = None
image_processor = None
if GEMINI_API_KEY:
    reasoning_system = ReasoningSystem(GEMINI_API_KEY, model_name="gemini-2.0-flash")
    image_processor = ImageProcessor(GEMINI_API_KEY, model_name="gemini-2.0-flash")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Helper functions
def get_conversation(db: Session, conversation_id, user_id):
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation

def get_user_config(db: Session, user_id):
    config = db.query(Configuration).filter(Configuration.user_id == user_id).first()
    if not config:
        # Create default config
        config = Configuration(user_id=user_id)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config

def save_base64_image(image_data: str, user_id: str):
    """Save a base64 encoded image to disk and return the file path"""
    try:
        # Remove the data:image/jpeg;base64, prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Generate a unique filename
        unique_filename = f"{user_id}_{uuid.uuid4()}.jpg"
        file_path = os.path.join("uploads", unique_filename)
        
        # Save the image
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        return file_path
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None

async def process_message_with_updates(
    user_message_id: str,
    query: str,
    image_path: Optional[str],
    chat_history,
    processor_config,
    db: Session,
    conversation_id: str,
    user_id: str
):
    """Process a message with real-time updates via WebSocket"""
    try:
        # Initialize reasoning system if not already initialized
        global reasoning_system, image_processor
        if reasoning_system is None:
            reasoning_system = ReasoningSystem(GEMINI_API_KEY, model_name="gemini-2.0-flash")
        if image_processor is None and GEMINI_API_KEY:
            image_processor = ImageProcessor(GEMINI_API_KEY, model_name="gemini-2.0-flash")
        
        # Configure model
        model = reasoning_system.configure_model(
            temperature=processor_config.get("temperature", 0.7),
            top_p=processor_config.get("top_p", 0.95),
            top_k=processor_config.get("top_k", 64),
            max_output_tokens=processor_config.get("max_output_tokens", 4096)
        )
        
        # Initialize thinking steps
        thinking_steps = []
        
        # If there's an image, process it first
        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                
                # Combine image analysis with the query
                image_analysis = image_processor.analyze_image(image_data, query, processor_config)
                query = f"{query}\n\nImage Analysis: {image_analysis}"
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                # Continue with just the text query if image processing fails
        
        # Step 1: Analyze the query
        analysis = reasoning_system.analyze_query(model, query, chat_history)
        
        # Save analysis step to database
        analysis_step = ReasoningStep(
            message_id=user_message_id,
            step_number=0,
            content=analysis,
            type="analysis"
        )
        db.add(analysis_step)
        db.commit()
        db.refresh(analysis_step)
        
        # Send update via WebSocket
        if user_id in active_connections:
            await active_connections[user_id].send_json({
                "type": "thinking_step",
                "data": {
                    "message_id": user_message_id,
                    "step_number": 0,
                    "content": analysis,
                    "type": "analysis",
                    "is_complete": False
                }
            })
        
        thinking_steps.append({
            "step_number": 0,
            "type": "analysis",
            "content": analysis
        })
        
        # Current query starts as the original
        current_query = query
        
        # Step 2: Iterative reasoning (if enabled)
        if processor_config.get("enable_reasoning", True):
            max_iter = processor_config.get("max_iterations", 3)
            
            for i in range(max_iter):
                # Improve the prompt with each iteration
                reasoning_output = reasoning_system.improve_prompt(
                    model, 
                    current_query, 
                    i, 
                    max_iter,
                    analysis,
                    chat_history
                )
                
                # Save reasoning step to database
                reasoning_step = ReasoningStep(
                    message_id=user_message_id,
                    step_number=i+1,
                    content=reasoning_output,
                    type="iteration"
                )
                db.add(reasoning_step)
                db.commit()
                db.refresh(reasoning_step)
                
                # Send update via WebSocket
                if user_id in active_connections:
                    await active_connections[user_id].send_json({
                        "type": "thinking_step",
                        "data": {
                            "message_id": user_message_id,
                            "step_number": i+1,
                            "content": reasoning_output,
                            "type": "iteration",
                            "is_complete": False
                        }
                    })
                
                thinking_steps.append({
                    "step_number": i+1,
                    "type": "iteration",
                    "content": reasoning_output
                })
                
                # Extract the improved prompt
                if "IMPROVED PROMPT:" in reasoning_output:
                    current_query = reasoning_output.split("IMPROVED PROMPT:")[1].strip()
        
        # Step 3: Generate final response
        final_response = reasoning_system.generate_final_response(
            model, 
            current_query,
            analysis,
            chat_history
        )
        
        # Add assistant response to database
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=final_response
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        
        # Send final response via WebSocket
        if user_id in active_connections:
            await active_connections[user_id].send_json({
                "type": "final_response",
                "data": {
                    "message_id": assistant_message.id,
                    "content": final_response,
                    "timestamp": assistant_message.timestamp.isoformat(),
                    "is_complete": True
                }
            })
        
        return {
            "response": final_response,
            "thinking_steps": thinking_steps,
            "assistant_message_id": assistant_message.id
        }
    except Exception as e:
        # Log the error
        print(f"Error in process_message_with_updates: {str(e)}")
        
        # Send error via WebSocket
        if user_id in active_connections:
            await active_connections[user_id].send_json({
                "type": "error",
                "data": {
                    "message": f"An error occurred: {str(e)}"
                }
            })
        
        # Add error message to database
        error_message = f"An error occurred: {str(e)}"
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=error_message
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        
        return {
            "response": error_message,
            "thinking_steps": [],
            "assistant_message_id": assistant_message.id
        }

# WebSocket endpoint for real-time updates
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    active_connections[user_id] = websocket
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            # Echo back to confirm connection is active
            await websocket.send_json({"status": "connected", "message": "Connection active"})
    except WebSocketDisconnect:
        if user_id in active_connections:
            del active_connections[user_id]

# Endpoints
# Update the send_message function to handle image_data properly
@router.post("/message", response_model=MessageResponse)
async def send_message(
    background_tasks: BackgroundTasks,
    message: MessageCreate,
    conversation_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if Gemini API key is configured
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API key not configured"
        )
    
    try:
        # Initialize reasoning system if not already initialized
        global reasoning_system
        if reasoning_system is None:
            reasoning_system = ReasoningSystem(GEMINI_API_KEY, model_name="gemini-2.0-flash")
        
        # Process image if provided
        image_path = None
        if message.image_data and message.image_data != "string":
            image_path = save_base64_image(message.image_data, current_user.id)
        
        # Get or create conversation
        if conversation_id:
            conversation = get_conversation(db, conversation_id, current_user.id)
        else:
            # Create new conversation with default title (first 30 chars of message)
            title = message.content[:30] + "..." if len(message.content) > 30 else message.content
            conversation = Conversation(
                user_id=current_user.id,
                title=title
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        
        # Add user message to database
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=message.content,
            image_url=image_path
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        
        # Get user configuration
        config = get_user_config(db, current_user.id)
        
        # Get conversation history
        chat_history = db.query(Message).filter(
            Message.conversation_id == conversation.id
        ).order_by(Message.timestamp).all()
        
        # Process message in background
        processor_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_output_tokens,
            "enable_reasoning": config.enable_reasoning,
            "max_iterations": config.max_iterations
        }
        
        # Start processing in background
        background_tasks.add_task(
            process_message_with_updates,
            user_message.id,
            message.content,
            image_path,
            chat_history,
            processor_config,
            db,
            conversation.id,
            current_user.id
        )
        
        # Update conversation title if this is the first message
        if len(chat_history) <= 1:
            conversation.title = message.content[:30] + "..." if len(message.content) > 30 else message.content
            db.commit()
        
        # Get any reasoning steps that might have been created already
        reasoning_steps = db.query(ReasoningStep).filter(
            ReasoningStep.message_id == user_message.id
        ).order_by(ReasoningStep.step_number).all()
        
        # Create response with reasoning steps
        response = {
            "id": user_message.id,
            "role": user_message.role,
            "content": user_message.content,
            "timestamp": user_message.timestamp,
            "image_url": user_message.image_url,
            "reasoning_steps": [
                {
                    "id": step.id,
                    "step_number": step.step_number,
                    "content": step.content,
                    "type": step.type
                } for step in reasoning_steps
            ]
        }
        
        # Return the user message with reasoning steps
        return response
        
    except Exception as e:
        # Handle errors
        error_message = f"An error occurred: {str(e)}"
        
        # Log the error
        print(f"Error in send_message: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )

@router.get("/conversations", response_model=List[ConversationListResponse])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id
        ).order_by(Conversation.updated_at.desc()).all()
        
        result = []
        for conv in conversations:
            # Get last message for preview
            last_message = db.query(Message).filter(
                Message.conversation_id == conv.id
            ).order_by(Message.timestamp.desc()).first()
            
            last_message_content = None
            if last_message:
                # Truncate message for preview
                last_message_content = last_message.content[:50] + "..." if len(last_message.content) > 50 else last_message.content
            
            result.append({
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "last_message": last_message_content
            })
        
        return result
    except Exception as e:
        # Log the error
        print(f"Error in get_conversations: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversations: {str(e)}"
        )

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_by_id(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        conversation = get_conversation(db, conversation_id, current_user.id)
        
        # Explicitly load messages to avoid lazy loading issues
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).all()
        
        # Create response object manually to ensure proper structure
        response = {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": []
        }
        
        for msg in messages:
            message_dict = {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "image_url": msg.image_url,
                "reasoning_steps": []
            }
            
            # Add reasoning steps if this is a user message
            if msg.role == "user":
                reasoning_steps = db.query(ReasoningStep).filter(
                    ReasoningStep.message_id == msg.id
                ).order_by(ReasoningStep.step_number).all()
                
                for step in reasoning_steps:
                    message_dict["reasoning_steps"].append({
                        "id": step.id,
                        "step_number": step.step_number,
                        "content": step.content,
                        "type": step.type
                    })
            
            response["messages"].append(message_dict)
        
        return response
    except Exception as e:
        # Log the error
        print(f"Error in get_conversation_by_id: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversation: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        conversation = get_conversation(db, conversation_id, current_user.id)
        
        # Delete all messages and reasoning steps first
        messages = db.query(Message).filter(Message.conversation_id == conversation_id).all()
        for message in messages:
            # Delete reasoning steps
            db.query(ReasoningStep).filter(ReasoningStep.message_id == message.id).delete()
            
            # Delete image if exists
            if message.image_url and os.path.exists(message.image_url):
                try:
                    os.remove(message.image_url)
                except Exception as e:
                    print(f"Error deleting image {message.image_url}: {str(e)}")
        
        # Delete all messages
        db.query(Message).filter(Message.conversation_id == conversation_id).delete()
        
        # Now delete the conversation
        db.delete(conversation)
        db.commit()
        
        return {"status": "success", "message": "Conversation deleted successfully"}
    except Exception as e:
        # Log the error
        print(f"Error in delete_conversation: {str(e)}")
        
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}"
        )

@router.get("/reasoning/{message_id}", response_model=List[ReasoningStepResponse])
async def get_reasoning_steps(
    message_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify the message belongs to the user
        message = db.query(Message).join(Conversation).filter(
            Message.id == message_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )
        
        # Get reasoning steps
        reasoning_steps = db.query(ReasoningStep).filter(
            ReasoningStep.message_id == message_id
        ).order_by(ReasoningStep.step_number).all()
        
        return reasoning_steps
    except Exception as e:
        # Log the error
        print(f"Error in get_reasoning_steps: {str(e)}")
        
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving reasoning steps: {str(e)}"
        )

