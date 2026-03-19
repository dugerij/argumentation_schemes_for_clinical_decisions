import os

def startup_check():
    """"Checks that all required environment variables are set and valid."""
    required_env_vars = [
        'GENERATOR_PROVIDER',
        'VERIFIER_PROVIDER',
        'REASONER_PROVIDER',
        'GENERATOR_BASE_MODEL',
        'VERIFIER_BASE_MODEL',
        'REASONER_BASE_MODEL'
    ]
    
    for var in required_env_vars:
        if var not in os.environ:
            raise ValueError(f"Missing required environment variable: {var}")
            
    if (os.environ.get('GENERATOR_PROVIDER') and os.environ.get('VERIFIER_PROVIDER') and os.environ.get('REASONER_PROVIDER')) not in ['ollama', 'openai', 'gemini']:
        raise ValueError("Invalid provider specified. Please choose from 'ollama', 'openai', or 'gemini'.")

        
    for var in required_env_vars:
        if 'PROVIDER' in var:
            provider = os.environ.get(var)
            if provider == 'ollama' and not os.environ.get('OLLAMA_MODEL'):
                raise ValueError("OLLAMA_MODEL must be set when using Ollama as a provider.")
            elif provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY must be set when using OpenAI as a provider.")
            elif provider == 'gemini' and not os.environ.get('GEMINI_API_KEY'):
                raise ValueError("GEMINI_API_KEY must be set when using Gemini as a provider.")
            
    print("Startup check passed. All required environment variables are set and valid.")