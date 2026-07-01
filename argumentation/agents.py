import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Optional

from helpers.jsonl import JsonlLogger
from helpers.config import get_model_name
from helpers.ollama import ollama_chat
from helpers.paths import EVENT_LOG_PATH

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ArgMed")

# ==========================================
# Data Structures
# ==========================================

@dataclass
class DialogueTurn:
    """Stores the interaction state for a single round of dialogue."""
    round_num: int
    generator_argument: str
    verifier_status: str
    verifier_cq: Optional[str] = None

# ==========================================
# Agent Classes
# ==========================================

class GeneratorAgent:
    def __init__(self, model: str):
        self.model = model
        logger.info(f"Initialized GeneratorAgent with model: {self.model}")

    def generate(self, question: str, rag_context: str, previous_critique: str = None) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a medical reasoning generator. "
                "Given a question and medical context, provide a decision and the supporting beliefs. "
                "Format your response clearly separating: \n"
                "DECISION (Args_d): [Your choice]\n"
                "BELIEFS (Args_b): [Your medical reasoning]"
            )}
        ]
        
        prompt = f"Context: {rag_context}\n\nQuestion: {question}\n"
        if previous_critique:
            prompt += f"\nVERIFIER CRITIQUE TO ADDRESS (cq): {previous_critique}\nUpdate your argument or defend it with new beliefs."
            
        messages.append({"role": "user", "content": prompt})
        
        logger.info("Generator is formulating an argument...")
        return ollama_chat(model=self.model, messages=messages)

class VerifierAgent:
    def __init__(self, model: str):
        self.model = model
        logger.info(f"Initialized VerifierAgent with model: {self.model}")

    def verify(self, question: str, argument: str) -> dict:
        messages = [
            {"role": "system", "content": (
                "You are a medical critical verifier. Evaluate the provided medical argument. "
                "Check for logical flaws, contradictions with context, or contraindications. "
                "Respond ONLY in valid JSON format: "
                '{"status": "ACCEPT" or "REJECT", "critical_question": "If rejecting, ask a specific question targeting the flaw, otherwise null"}'
            )},
            {"role": "user", "content": f"Question: {question}\n\nArgument to Verify:\n{argument}"}
        ]
        
        logger.info("Verifier is evaluating the argument...")
        content = ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)

class ReasonerAgent:
    def __init__(self, model: str):
        self.model = model
        logger.info(f"Initialized ReasonerAgent with model: {self.model}")

    def reason(self, dialogue_history: List[DialogueTurn]) -> str:
        messages = [
            {"role": "system", "content": (
                "You are the Argumentation Framework Reasoner. Review the dialogue between the Generator and Verifier. "
                "Your task is to summarize the 'Explanation Set' (Definition 4): "
                "1. State the final accepted Optional Decision. "
                "2. Provide the maximal, conflict-free set of acceptable beliefs that support this decision. "
                "Ignore rejected arguments and resolve the final framework."
            )}
        ]
        
        history_text = ""
        for turn in dialogue_history:
            history_text += (
                f"Turn {turn.round_num}:\n"
                f"Generator: {turn.generator_argument}\n"
                f"Verifier Status: {turn.verifier_status}\n"
                f"Verifier Critique: {turn.verifier_cq}\n\n"
            )
            
        messages.append({"role": "user", "content": history_text})
        
        logger.info("Reasoner is assessing the final framework...")
        return ollama_chat(model=self.model, messages=messages)

# ==========================================
# Orchestrator Class
# ==========================================

class ArgumentInteraction:
    """Orchestrates the dialogue between the models and stores the state."""
    
    def __init__(
        self,
        question: str,
        rag_context: str,
        max_rounds: int,
        event_logger: JsonlLogger | None = None,
    ):
        self.question = question
        self.rag_context = rag_context
        self.max_rounds = max_rounds
        self.dialogue_history: List[DialogueTurn] = []
        self.event_logger = event_logger or JsonlLogger(EVENT_LOG_PATH, run_id="argument_interaction")
        
        self.generator = GeneratorAgent(get_model_name("GENERATOR"))
        self.verifier = VerifierAgent(get_model_name("VERIFIER"))
        self.reasoner = ReasonerAgent(get_model_name("REASONER"))

    def run(self) -> str:
        logger.info("Starting ArgMed Pipeline Interaction")
        self.event_logger.event(
            "argument_interaction",
            "started",
            max_rounds=self.max_rounds,
            question_preview=self.question[:500],
            context_chars=len(self.rag_context),
        )
        previous_critique = None
        
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"--- Beginning Round {round_num} ---")
            self.event_logger.event("argument_round", "started", round_num=round_num)
            
            # Generator turn
            with self.event_logger.timed("generator", round_num=round_num):
                argument = self.generator.generate(self.question, self.rag_context, previous_critique)
            logger.info(f"GENERATOR OUTPUT:\n{argument}")
            self.event_logger.event(
                "generator",
                "output",
                round_num=round_num,
                argument_preview=argument[:2000],
                argument_chars=len(argument),
            )
            
            # Verifier turn
            with self.event_logger.timed("verifier", round_num=round_num):
                verification = self.verifier.verify(self.question, argument)
            status = verification.get("status", "REJECT").upper()
            cq = verification.get("critical_question")
            logger.info(f"VERIFIER STATUS: {status} | CQ: {cq}")
            self.event_logger.event(
                "verifier",
                "output",
                round_num=round_num,
                verifier_status=status,
                critical_question=cq,
                raw_verification=verification,
            )
            
            # Record state
            turn = DialogueTurn(
                round_num=round_num,
                generator_argument=argument,
                verifier_status=status,
                verifier_cq=cq
            )
            self.dialogue_history.append(turn)
            
            # Check loop condition
            if status == "ACCEPT":
                logger.info("Argument accepted. Ending dialogue loop.")
                self.event_logger.event("argument_round", "completed", round_num=round_num, accepted=True)
                break
            else:
                previous_critique = cq
                self.event_logger.event("argument_round", "completed", round_num=round_num, accepted=False)
                if round_num == self.max_rounds:
                    logger.warning("Max rounds reached with unresolved conflicts.")
                    self.event_logger.event("argument_interaction", "max_rounds_reached", max_rounds=self.max_rounds)
                    
        # Reasoner turn
        logger.info("--- Handing over to Reasoner ---")
        with self.event_logger.timed("reasoner", turns=len(self.dialogue_history)):
            final_assessment = self.reasoner.reason(self.dialogue_history)
        logger.info(f"FINAL ASSESSMENT:\n{final_assessment}")
        self.event_logger.event(
            "argument_interaction",
            "completed",
            turns=len(self.dialogue_history),
            final_assessment_preview=final_assessment[:2000],
            final_assessment_chars=len(final_assessment),
        )
        
        return final_assessment

    def dump_history(self) -> str:
        """Utility to export the dialogue history as JSON."""
        return json.dumps([asdict(turn) for turn in self.dialogue_history], indent=2)

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Get config from env
    MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", 3))

    medqa_question = (
        "An 18-year-old woman presents with recurrent headaches. The pain is usually unilateral, "
        "pulsatile in character, exacerbated by light and noise, and usually lasts for a few hours "
        "to a full day. The pain is sometimes triggered by eating chocolates. These headaches disturb "
        "her daily routine activities. The physical examination was within normal limits. She also has "
        "essential tremors. Which drug is suitable in her case for the prevention of headaches?"
    )
    
    mock_rag_context = (
        "1. Migraine presentation: unilateral, pulsatile, photophobia, phonophobia. "
        "2. Prophylactic migraine drugs: Beta-blockers (Propranolol), Anticonvulsants (Topiramate, Valproate), Calcium channel blockers, Amitriptyline. "
        "3. Essential tremors treatment: Propranolol is a first-line agent. "
        "4. Topiramate can cause cognitive dulling and weight loss, but does not treat essential tremors."
    )
    
    # Initialize the interaction session
    interaction = ArgumentInteraction(
        question=medqa_question,
        rag_context=mock_rag_context,
        max_rounds=MAX_ROUNDS
    )
    
    # Run the pipeline
    final_output = interaction.run()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE. FINAL REASONER OUTPUT:")
    print("="*50)
    print(final_output)
    
    # If you need to save the dialogue history to a file:
    # with open("dialogue_log.json", "w") as f:
    #     f.write(interaction.dump_history())
