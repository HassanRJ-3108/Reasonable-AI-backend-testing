"""
Reasoning System for Reasonable AI

This module implements the multi-step reasoning process that makes Reasonable AI unique.
"""

import google.generativeai as genai
from ai.prompts import get_analysis_prompt, get_reasoning_prompt, get_response_prompt
from typing import Optional
from sqlalchemy.orm import Session
from ai.image_processor import ImageProcessor
from models import ReasoningStep, Message
import os


# Load environment variables
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize reasoning system and image processor (global scope)
reasoning_system = None
image_processor = None
active_connections = {}  # Placeholder for WebSocket connections

class ReasoningSystem:
    """
    Implements the multi-step reasoning process for Reasonable AI.
    This class handles the core logic of analyzing queries, performing
    iterative reasoning, and generating final responses.
    """
    
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        Initialize the reasoning system
        
        Args:
            api_key (str): Gemini API key
            model_name (str): Name of the Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
    
    def configure_model(self, temperature=0.7, top_p=0.95, top_k=64, max_output_tokens=4096):
        """
        Configure the Gemini model parameters
        
        Args:
            temperature (float): Controls randomness (0.0 to 1.0)
            top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0)
            top_k (int): Controls diversity via top-k sampling
            max_output_tokens (int): Maximum number of tokens in the response
            
        Returns:
            GenerativeModel: Configured Gemini model
        """
        model_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        return genai.GenerativeModel(self.model_name, generation_config=model_params)
    
    def analyze_query(self, model, query, conversation_history):
        """
        Analyze the user's query to understand intent and context
        
        Args:
            model: Configured Gemini model
            query (str): The user's query
            conversation_history (list): Previous messages
            
        Returns:
            str: Analysis of the query
        """
        prompt = get_analysis_prompt(query, conversation_history)
        response = model.generate_content(prompt)
        return response.text
    
    def improve_prompt(self, model, original_query, iteration, max_iterations, analysis, conversation_history):
        """
        Improve the prompt through iterative reasoning
        
        Args:
            model: Configured Gemini model
            original_query (str): The user's original query
            iteration (int): Current iteration number
            max_iterations (int): Maximum number of iterations
            analysis (str): Initial analysis
            conversation_history (list): Previous messages
            
        Returns:
            str: Improved prompt
        """
        prompt = get_reasoning_prompt(
            original_query, 
            iteration, 
            max_iterations, 
            analysis, 
            conversation_history
        )
        response = model.generate_content(prompt)
        return response.text
    
    def generate_final_response(self, model, improved_prompt, analysis, conversation_history):
        """
        Generate the final response based on the improved prompt
        
        Args:
            model: Configured Gemini model
            improved_prompt (str): The improved prompt
            analysis (str): Initial analysis
            conversation_history (list): Previous messages
            
        Returns:
            str: Final response
        """
        prompt = get_response_prompt(improved_prompt, analysis, conversation_history)
        response = model.generate_content(prompt)
        return response.text
    
    def process_message(self, query, conversation_history, config):
        """
        Process a user message using the multi-step reasoning approach
        
        Args:
            query (str): The user's message
            conversation_history (list): Previous messages
            config (dict): Configuration parameters
            
        Returns:
            dict: Contains final response and thinking steps
        """
        # Configure model based on user preferences
        model = self.configure_model(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 64),
            max_output_tokens=config.get("max_output_tokens", 4096)
        )
        
        # Initialize thinking steps
        thinking_steps = []
        
        # Step 1: Analyze the query
        analysis = self.analyze_query(model, query, conversation_history)
        thinking_steps.append({
            "step_number": 0,
            "type": "analysis",
            "content": analysis
        })
        
        # Current query starts as the original
        current_query = query
        
        # Step 2: Iterative reasoning (if enabled)
        if config.get("enable_reasoning", True):
            max_iter = config.get("max_iterations", 3)
            
            for i in range(max_iter):
                # Improve the prompt with each iteration
                reasoning_output = self.improve_prompt(
                    model, 
                    current_query, 
                    i, 
                    max_iter,
                    analysis,
                    conversation_history
                )
                
                thinking_steps.append({
                    "step_number": i+1,
                    "type": "iteration",
                    "content": reasoning_output
                })
                
                # Extract the improved prompt
                if "IMPROVED PROMPT:" in reasoning_output:
                    current_query = reasoning_output.split("IMPROVED PROMPT:")[1].strip()
        
        # Step 3: Generate final response
        final_response = self.generate_final_response(
            model, 
            current_query,
            analysis,
            conversation_history
        )
        
        # Return response and thinking steps
        return {
            "response": final_response,
            "thinking_steps": thinking_steps
        }

# Update the process_message_with_updates function to handle image URLs
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
                # Check if it's a URL or a local file path
                if image_path.startswith(('http://', 'https://')):
                    # It's a URL, download the image
                    import requests
                    response = requests.get(image_path)
                    image_data = response.content
                else:
                    # It's a local file path
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

