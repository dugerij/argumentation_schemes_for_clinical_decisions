import os
import json

def startup_check():
    """"Checks that all required environment variables are set and valid
    Raises:
        ValueError: If any required environment variable is missing or invalid.
    """
    required_env_vars = [
        'GENERATION_MODEL_PROVIDER',
        'VERIFIER_MODEL_PROVIDER',
        'REASONER_MODEL_PROVIDER',
        'GENERATOR_BASE_MODEL',
        'VERIFIER_BASE_MODEL',
        'REASONER_BASE_MODEL'
    ]
    
    for var in required_env_vars:
        if var not in os.environ:
            raise ValueError(f"Missing required environment variable: {var}")
            
    # Validate providers are valid values
    providers = [
        os.environ.get('GENERATION_MODEL_PROVIDER'),
        os.environ.get('VERIFIER_MODEL_PROVIDER'),
        os.environ.get('REASONER_MODEL_PROVIDER')
    ]
    for provider in providers:
        if provider not in ['ollama', 'openai', 'gemini']:
            raise ValueError("Invalid provider specified. Please choose from 'ollama', 'openai', or 'gemini'.")

    # Validate provider-specific credentials
    provider_vars = [
        'GENERATION_MODEL_PROVIDER',
        'VERIFIER_MODEL_PROVIDER',
        'REASONER_MODEL_PROVIDER'
    ]
    for var in provider_vars:
        provider = os.environ.get(var)
        if provider == 'ollama' and not os.environ.get('OLLAMA_MODEL'):
            raise ValueError("OLLAMA_MODEL must be set when using Ollama as a provider.")
        elif provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI as a provider.")
        elif provider == 'gemini' and not os.environ.get('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY must be set when using Gemini as a provider.")
            
    print("Startup check passed. All required environment variables are set and valid.")


def load_dataset(file_path: str):
    with open(file_path, 'r') as file:
        for line in file:
            raw_json = json.loads(line)
            print(raw_json)