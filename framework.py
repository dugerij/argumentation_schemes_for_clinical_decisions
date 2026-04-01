from ollama import chat

class Generator(object):
    """LLM-based argument generator."""

    def __init__(self, name="default-generator", description="",
                 generator_type="ollama", generator_args=None, model="llama3.2:1b"):
        self.name = name
        self.description = description
        self.generator_type = generator_type
        self.generator_args = generator_args or {}
        self.model = model

    def get_argument(self, query: str) -> str:
        response = chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": query
            }],
            think=False,
            params={
                "presence_penalty": 1.5,
                "temperature": 0.0,
                "top_k": 20,
                "top_p": 0.95,
                **self.generator_args,
            }
        )
        return response.message.content


class Verifier(object):
    """Verifies generated arguments against ground truth and RAG context."""

    def __init__(self, name="default-verifier", description="", verifier_type="rule", verifier_args=None):
        self.name = name
        self.description = description
        self.verifier_type = verifier_type
        self.verifier_args = verifier_args or {}

    def verify_argument(self, generated_answer: str, gold_answer: str, rag_context: str = "") -> dict:
        satisfied = generated_answer.strip().lower() == str(gold_answer).strip().lower()
        merged = (generated_answer + " " + rag_context).strip().lower()

        # basic rag-awareness heuristic
        if not satisfied and gold_answer.lower() in merged:
            satisfied = True

        return {
            "generated_answer": generated_answer,
            "gold_answer": gold_answer,
            "satisfied": satisfied,
            "confidence": 1.0 if satisfied else 0.0,
            "context_snippet": rag_context,
        }


class Reasoner(object):
    """Provides final soundness check of the argument discussion."""

    def __init__(self, name="default-reasoner", description="", reasoner_type="rule", reasoner_args=None):
        self.name = name
        self.description = description
        self.reasoner_type = reasoner_type
        self.reasoner_args = reasoner_args or {}

    def check(self, discussion_history: list) -> dict:
        sound = all(item.get("satisfied", True) for item in discussion_history if isinstance(item, dict))

        return {
            "sound": sound,
            "summary": "The pipeline produced a sound argument sequence" if sound else "The pipeline found inconsistencies",
            "steps": discussion_history,
        }


class SessionManager(object):
    def __init__(self):
        self.state = {}
        self.active = False

    def start(self):
        self.state = {}
        self.active = True
        return self.state

    def end(self):
        self.active = False
        self.state = {}

    def is_active(self):
        return self.active