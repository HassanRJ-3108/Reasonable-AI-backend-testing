"""
Image Processing Module for Reasonable AI

This module handles image analysis using the Gemini model.
"""

import google.generativeai as genai
from PIL import Image
import io
from ai.prompts import get_image_analysis_prompt

class ImageProcessor:
    """
    Handles image analysis using the Gemini model.
    This class makes it easy to process images with the AI.
    """
    
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        Initialize the image processor
        
        Args:
            api_key (str): Gemini API key
            model_name (str): Name of the Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
    
    def configure_model(self, temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192):
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
    
    def analyze_image(self, image_data, prompt, config=None):
        """
        Analyze an image using the Gemini model
        
        Args:
            image_data (bytes): Raw image data
            prompt (str): User's prompt for image analysis
            config (dict, optional): Configuration parameters
            
        Returns:
            str: Analysis of the image
        """
        if config is None:
            config = {}
        
        # Configure model
        model = self.configure_model(
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 64),
            max_output_tokens=config.get("max_output_tokens", 8192)
        )
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        
        # Generate enhanced prompt
        enhanced_prompt = get_image_analysis_prompt(prompt)
        
        # Generate content
        response = model.generate_content([enhanced_prompt, image])
        
        return response.text

