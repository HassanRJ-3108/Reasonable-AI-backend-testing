from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv

from database import get_db
from models import User, Conversation, Message, ReasoningStep, Configuration
from schemas import (
    ConversationCreate, 
    ConversationResponse, 
    ConversationListResponse,
    MessageCreate, 
    MessageResponse,
    ReasoningStepResponse
)
from auth import get_current_user

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
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

def analyze_query(model, query, chat_history):
    # Format chat history for context
    formatted_history = ""
    for msg in chat_history[-5:]:  # Use last 5 messages for context
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    
    analysis_prompt = f"""
    You are Reasonable AI, created by Hassan RJ - Full Stack Nextjs Developer | Python developer | Student Leader | Web 3.0 | Generative AI | Agentic AI | AI Assistants | AI Powered Chatbots | Building Dynamic Web Apps | Learning & Growing!
    
    TASK: Analyze the user's query to understand what they're asking for.
    
    Recent conversation context:
    {formatted_history}
    
    Current query: "{query}"
    
    Analyze this query and determine:
    1. What is the user asking for?
    2. How detailed should the response be?
    3. What specific information should be included in the response?
    4. How should previous context be considered?
    
    Format your response as follows:
    QUERY_INTENT: [Brief description of what the user is asking for]
    DESIRED_DETAIL_LEVEL: [BRIEF, MODERATE, DETAILED]
    KEY_POINTS_TO_ADDRESS: [List the main points that should be addressed in the response]
    CONTEXT_CONSIDERATION: [How previous messages should influence the response]
    ANALYSIS: [Your detailed analysis of the query]
    """
    
    response = model.generate_content(analysis_prompt)
    return response.text

def improve_prompt(model, original_query, iteration, max_iterations, chat_history, analysis):
    # Format chat history for context
    formatted_history = ""
    for msg in chat_history[-5:]:  # Use last 5 messages for context
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    
    system_prompt = f"""
    You are an expert prompt engineer. Your task is to analyze and improve the following query.
    
    Recent conversation context:
    {formatted_history}
    
    Initial analysis:
    {analysis}
    
    Iteration: {iteration+1} of {max_iterations}
    
    Current query: "{original_query}"
    
    Your task:
    1. Analyze the query to understand the user's intent, considering the conversation context
    2. Identify any ambiguities or missing context
    3. Restructure and enhance the query to get the best possible response
    4. Make the prompt more specific, detailed, and clear
    5. Ensure the improved prompt will generate a substantial, detailed response
    
    Format your response as follows:
    ANALYSIS: [Your detailed analysis of the query]
    IMPROVED PROMPT: [The improved prompt]
    """
    
    response = model.generate_content(system_prompt)
    return response.text

def generate_final_response(model, improved_prompt, chat_history, analysis):
    # Format chat history for context
    formatted_history = ""
    for msg in chat_history[-5:]:  # Use last 5 messages for context
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    
    # Extract detail level from analysis
    detail_level = "DETAILED"
    if "DESIRED_DETAIL_LEVEL:" in analysis:
        if "BRIEF" in analysis:
            detail_level = "BRIEF"
        elif "MODERATE" in analysis:
            detail_level = "MODERATE"
    
    # Add context to the improved prompt
    prompt_with_context = f"""
    You are Reasonable AI, created by Hassan RJ - Full Stack Nextjs Developer | Python developer | Student Leader | Web 3.0 | Generative AI | Agentic AI | AI Assistants | AI Powered Chatbots | Building Dynamic Web Apps | Learning & Growing!
    
    Recent conversation context:
    {formatted_history}
    
    Initial analysis:
    {analysis}
    
    Based on the above context and analysis, please respond to the following:
    {improved_prompt}
    
    IMPORTANT INSTRUCTIONS:
    1. Provide a {detail_level.lower()} response that directly addresses the query
    2. Even for simple greetings, provide a friendly, substantial response (not just "Hi" or "Hello")
    3. For questions, provide informative, well-structured answers
    4. Make sure your response is relevant to what the user is asking
    5. Avoid unnecessary information that doesn't relate to the query
    6. If the user is asking for a greeting, respond with a warm, friendly greeting
    7. If the user is acknowledging something, provide a meaningful acknowledgment
    8. NEVER mention that you are powered by Gemini or any other AI model
    """
    
    response = model.generate_content(prompt_with_context)
    return response.text

def configure_genai_model(config):
    model_params = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_output_tokens": config.max_output_tokens,
    }
    return genai.GenerativeModel(config.model, generation_config=model_params)

# Endpoints
@router.post("/message", response_model=MessageResponse)
async def send_message(
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
        image_url=message.image_url
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    # Get user configuration
    config = get_user_config(db, current_user.id)
    
    # Configure Gemini model
    model = configure_genai_model(config)
    
    # Get conversation history
    chat_history = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.timestamp).all()
    
    # Initialize thinking steps
    thinking_steps = []
    
    try:
        # First API call: Analyze the query
        analysis = analyze_query(model, message.content, chat_history)
        analysis_step = ReasoningStep(
            message_id=user_message.id,
            step_number=0,
            content=analysis,
            type="analysis"
        )
        thinking_steps.append(analysis_step)
        
        # Iterative reasoning process
        current_query = message.content
        
        # Perform reasoning iterations if enabled
        if config.enable_reasoning:
            for i in range(config.max_iterations):
                # Improve the prompt with context and analysis
                reasoning_output = improve_prompt(
                    model, 
                    current_query, 
                    i, 
                    config.max_iterations,
                    chat_history,
                    analysis
                )
                
                iteration_step = ReasoningStep(
                    message_id=user_message.id,
                    step_number=i+1,
                    content=reasoning_output,
                    type="iteration"
                )
                thinking_steps.append(iteration_step)
                
                # Extract the improved prompt
                if "IMPROVED PROMPT:" in reasoning_output:
                    current_query = reasoning_output.split("IMPROVED PROMPT:")[1].strip()
        
        # Generate final response
        final_response = generate_final_response(
            model, 
            current_query,
            chat_history,
            analysis
        )
        
        # Add assistant response to database
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=final_response
        )
        db.add(assistant_message)
        
        # Add thinking steps to database
        for step in thinking_steps:
            db.add(step)
        
        db.commit()
        db.refresh(assistant_message)
        
        # Update conversation title if this is the first message
        if len(chat_history) <= 1:
            conversation.title = message.content[:30] + "..." if len(message.content) > 30 else message.content
            db.commit()
        
        return assistant_message
        
    except Exception as e:
        # Handle errors
        error_message = f"An error occurred: {str(e)}"
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=error_message
        )
        db.add(assistant_message)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )

@router.get("/conversations", response_model=List[ConversationListResponse])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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

@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = get_conversation(db, conversation_id, current_user.id)
    return conversation

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = get_conversation(db, conversation_id, current_user.id)
    db.delete(conversation)
    db.commit()
    return {"status": "success"}

@router.get("/reasoning/{message_id}", response_model=List[ReasoningStepResponse])
async def get_reasoning_steps(
    message_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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

