"""
Simple Agent framework for trading agents
"""

class ModelSettings:
    """Model settings for AI agents"""
    def __init__(self, model_name="gpt-4", temperature=0.3, max_tokens=2000, 
                 top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

class Agent:
    """Base Agent class"""
    def __init__(self, name="Agent"):
        self.name = name
        self.description = "Base Agent"
        self.tools = []
        self.model = None
        self.max_iterations = 10
        self.max_response_time = 300

def function_tool(func, name=None, description=None):
    """Create a function tool wrapper"""
    class ToolWrapper:
        def __init__(self, func, name, description):
            self.func = func
            self.name = name or func.__name__
            self.description = description or func.__doc__ or "No description available"
            
        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)
    
    return ToolWrapper(func, name, description)