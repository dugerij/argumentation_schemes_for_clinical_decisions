class Generator(object):
    def __init__(self, name, description, generator_type, generator_args):
        self.name = name
        self.description = description
        self.generator_type = generator_type
        self.generator_args = generator_args

class Verifier(object):
    def __init__(self, name, description, verifier_type, verifier_args):
        self.name = name
        self.description = description
        self.verifier_type = verifier_type
        self.verifier_args = verifier_args

class Reasoner(object):
    def __init__(self, name, description, reasoner_type, reasoner_args):
        self.name = name
        self.description = description
        self.reasoner_type = reasoner_type
        self.reasoner_args = reasoner_args