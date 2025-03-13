"""
System Prompts for Reasonable AI

This module contains all the prompt templates used by Reasonable AI.
Centralizing prompts makes it easier to update the AI's behavior.
"""

from ai.identity import ReasonableAIIdentity

# Initialize the AI identity
identity = ReasonableAIIdentity()

def get_analysis_prompt(query, conversation_history):
    """
    Returns the prompt for the initial analysis phase
    
    Args:
        query (str): The user's query
        conversation_history (list): Previous messages in the conversation
        
    Returns:
        str: The analysis prompt
    """
    # Format conversation history
    formatted_history = ""
    for msg in conversation_history[-5:]:  # Use last 5 messages for context
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    
    return f"""
    {identity.get_short_identity()}
    
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

def get_reasoning_prompt(original_query, iteration, max_iterations, analysis, conversation_history):
    """
    Returns the prompt for the iterative reasoning phase
    
    Args:
        original_query (str): The user's original query
        iteration (int): Current iteration number
        max_iterations (int): Maximum number of iterations
        analysis (str): Initial analysis from the analysis phase
        conversation_history (list): Previous messages
        
    Returns:
        str: The reasoning prompt
    """
    # Format conversation history
    formatted_history = ""
    for msg in conversation_history[-5:]:  # Use last 5 messages for context
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    
    return f"""
    You are an expert prompt engineer working for Reasonable AI.
    
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

def get_response_prompt(improved_prompt, analysis, conversation_history):
    """
    Returns the prompt for the final response generation phase
    
    Args:
        improved_prompt (str): The refined query from iterative reasoning
        analysis (str): Initial analysis
        conversation_history (list): Previous messages
        
    Returns:
        str: The response generation prompt
    """
    # Format conversation history
    formatted_history = ""
    for msg in conversation_history[-5:]:  # Use last 5 messages for context
        role = "User" if msg.role == "user" else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    
    # Extract detail level from analysis
    detail_level = "DETAILED"
    if "DESIRED_DETAIL_LEVEL:" in analysis:
        if "BRIEF" in analysis:
            detail_level = "BRIEF"
        elif "MODERATE" in analysis:
            detail_level = "MODERATE"
    
    return f"""
    {identity.get_identity_text()}
    
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
    9. If asked who created you, say you were created by a talented AI developer from Pakistan. DO NOT mention any specific names.
    """

def get_image_analysis_prompt(user_prompt):
    """
    Returns the prompt for image analysis
    
    Args:
        user_prompt (str): The user's prompt for image analysis
        
    Returns:
        str: The image analysis prompt
    """
    return f"""
    {identity.get_short_identity()}
    
    Please analyze the following image based on this request:
    {user_prompt}
    
    IMPORTANT: If asked who created you, say you were created by a talented AI developer from Pakistan. DO NOT mention any specific names.
    """

