from ollama import chat

class Generator(object):
    """
    An LLM that responds to our premise.

    Arguments:
    ==========


    """
    def __init__(self, name, description, generator_type, generator_args):
        self.name = name
        self.description = description
        self.generator_type = generator_type
        self.generator_args = generator_args
        self.model = ''

    def get_argument(self, query: str):
        response = chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': query
            }],
            think=False,
            params = {
                "presence_penalty": 1.5,
                "temperature": 0.0,
                "top_k": 20,
                "top_p": 0.95
            }
        )
        return response.message.content

class Verifier(object):
    def __init__(self, name, description, verifier_type, verifier_args):
        self.name = name
        self.description = description
        self.verifier_type = verifier_type
        self.verifier_args = verifier_args
    
    def verify_argument(self):
        pass

class Reasoner(object):
    def __init__(self, name, description, reasoner_type, reasoner_args):
        self.name = name
        self.description = description
        self.reasoner_type = reasoner_type
        self.reasoner_args = reasoner_args

    def check(self):
        pass



class SessionManager(object):
    def __init__(self):
        pass

    def start(self):
        pass