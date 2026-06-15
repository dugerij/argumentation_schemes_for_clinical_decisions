from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Any

from entity_extraction.schema import ClinicalKnowledgeContext, UMLSConcept


# =====================================================================
# 1. SYMBOLIC ARGUMENT SCHEME CLASS (OVERRIDDEN HASH FOR DICT SAFETY)
# =====================================================================

@dataclass(frozen=True)
class Argument:
    id: str
    scheme_type: str               
    conclusion: str
    premises: Tuple[str, ...] = field(default_factory=tuple)  
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False) 

    # Override hashing to bypass mutable types (list/dict) in fields
    def __hash__(self):
        return hash(self.id)

    # Ensure equality checking aligns explicitly with your customized hash
    def __eq__(self, other):
        if not isinstance(other, Argument):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"{self.id}({self.scheme_type}: {self.conclusion})"


# =====================================================================
# 2. CLINICAL KNOWLEDGE TRANSLATION ENGINE
# =====================================================================

class ClinicalArgumentationEngine:
    def __init__(self, context: ClinicalKnowledgeContext):
        self.context = context
        self.arguments: Dict[str, Argument] = {}
        self.attacks: Set[Tuple[str, str]] = set()  
        self._arg_counter = 1

    def _next_id(self) -> str:
        id_str = f"Arg{self._arg_counter}"
        self._arg_counter += 1
        return id_str

    def generate_framework(self, proposed_treatments: List[str], goal: str) -> Tuple[Set[Argument], Set[Tuple[str, str]]]:
        self.arguments.clear()
        self.attacks.clear()
        self._arg_counter = 1

        aspt_arguments: Dict[str, str] = {}  

        for treatment in proposed_treatments:
            # --- ASPT ---
            aspt_id = self._next_id()
            aspt_arg = Argument(
                id=aspt_id,
                scheme_type="ASPT",
                conclusion=f"Consider treatment {treatment} to realize goal '{goal}'",
                premises=(  
                    f"Patient Facts: {', '.join(self.context.extracted_facts)}",
                    f"In order to realize the goal: {goal}",
                    f"Treatment {treatment} promotes the goal {goal}"
                ),
                metadata={"treatment": treatment, "goal": goal}
            )
            self.arguments[aspt_id] = aspt_arg
            aspt_arguments[treatment] = aspt_id

            # --- CQ1 (AS_Hist) ---
            for hist in self.context.past_failures_history:
                if hist['treatment'] == treatment and hist['failed_for_goal'] == goal:
                    cq1_id = self._next_id()
                    cq1_arg = Argument(
                        id=cq1_id,
                        scheme_type="AS_Hist",
                        conclusion=f"Treatment {treatment} is not effective for achieving goal '{goal}' based on medical history.",
                        premises=(  
                            f"Treatment {treatment} is identical/similar to a past treatment iteration.",
                            f"The past iteration was not effective in achieving goal '{goal}'."
                        ),
                        metadata={"treatment": treatment, "goal": goal}
                    )
                    self.arguments[cq1_id] = cq1_arg
                    self.attacks.add((cq1_id, aspt_id))  

            # --- CQ2 (AS_SideEffect) ---
            for se in self.context.side_effects_history:
                if se['treatment'] == treatment:
                    cq2_id = self._next_id()
                    cq2_arg = Argument(
                        id=cq2_id,
                        scheme_type="AS_SideEffect",
                        conclusion=f"Treatment {treatment} should not be considered due to harmful side effect: {se['effect']}",
                        premises=(  
                            f"{se['effect']} is an effect of treatment {treatment}.",
                            f"{se['effect']} is a harmful outcome relative to health goals."
                        ),
                        metadata={"treatment": treatment, "side_effect": se['effect']}
                    )
                    self.arguments[cq2_id] = cq2_arg
                    self.attacks.add((cq2_id, aspt_id))  

            # --- CQ3 (ASContraInd) ---
            for ci in self.context.contraindications_db:
                if ci['treatment'] == treatment and ci['condition'] in self.context.extracted_facts:
                    cq3_id = self._next_id()
                    cq3_arg = Argument(
                        id=cq3_id,
                        scheme_type="ASContraInd",
                        conclusion=f"Treatment {treatment} should not be used due to patient contraindication status.",
                        premises=(  
                            f"If treatment {treatment} is used when {ci['condition']} is true, then danger {ci['danger']} will occur.",
                            f"{ci['danger']} is a clinically dangerous risk state."
                        ),
                        metadata={"treatment": treatment, "condition": ci['condition']}
                    )
                    self.arguments[cq3_id] = cq3_arg
                    self.attacks.add((cq3_id, aspt_id))  

        # --- CQ4 ---
        t_names = list(aspt_arguments.keys())
        for i in range(len(t_names)):
            for j in range(i + 1, len(t_names)):
                id_a = aspt_arguments[t_names[i]]
                id_b = aspt_arguments[t_names[j]]
                self.attacks.add((id_a, id_b))
                self.attacks.add((id_b, id_a))

        return set(self.arguments.values()), self.attacks


# =====================================================================
# 3. PIPELINE EXECUTION
# =====================================================================

if __name__ == "__main__":
    from argumentation.aaf import AbstractArgumentationFramework

    print("================================================================")
    print("STEP 1: Emulating Knowledge Extraction via Graph RAG & UMLS")
    print("================================================================")
    
    patient_context = ClinicalKnowledgeContext(
        patient_id="MIMIC-IV-PT-10492",
        extracted_facts=["Hypertension", "Chronic Kidney Disease Stage 3"],
        past_failures_history=[
            {"treatment": "Drug_A", "failed_for_goal": "Lower patient blood pressure"}
        ],
        side_effects_history=[
            {"treatment": "Drug_B", "effect": "Severe Angioedema"}
        ],
        contraindications_db=[
            {"treatment": "Drug_B", "condition": "Chronic Kidney Disease Stage 3", "danger": "Acute Kidney Injury Progression"}
        ],
        umls_mappings={
            "Hypertension": UMLSConcept("C0020538", "Essential Hypertension", "Disease or Syndrome"),
            "Chronic Kidney Disease Stage 3": UMLSConcept("C1561643", "Chronic kidney disease stage 3", "Disease or Syndrome"),
            "Drug_A": UMLSConcept("C0021960", "Lisinopril", "Pharmacologic Substance"),
            "Drug_B": UMLSConcept("C0014431", "Enalapril", "Pharmacologic Substance"),
            "Drug_C": UMLSConcept("C0002773", "Amlodipine", "Pharmacologic Substance")
        }
    )

    print(f"Patient ID: {patient_context.patient_id}")
    print(f"Extracted Comorbidities: {patient_context.extracted_facts}")

    candidate_treatments = ["Drug_A", "Drug_B", "Drug_C"]
    target_health_goal = "Lower patient blood pressure"

    engine = ClinicalArgumentationEngine(context=patient_context)
    arguments, attack_relations = engine.generate_framework(candidate_treatments, target_health_goal)

    print("\n================================================================")
    print("STEP 2: Instantiating Formal Schemes & Critical Question Attacks")
    print("================================================================")
    print("\n[Generated Claims/Arguments]:")
    for arg in sorted(arguments, key=lambda x: x.id):
        print(f" * {arg.id:<6} [{arg.scheme_type:<14}] -> {arg.conclusion}")

    print("\n[Instantiated Attacks based on Rules]:")
    for attacker, attacked in sorted(attack_relations):
        print(f" * {attacker} attacks {attacked}")

    aaf_solver = AbstractArgumentationFramework(arguments, attack_relations)
    
    cf_sets = aaf_solver.compute_all_conflict_free_sets()
    admissible_sets = aaf_solver.compute_all_admissible_sets()
    grounded = aaf_solver.compute_grounded_extension()
    preferred = aaf_solver.compute_preferred_extensions()

    print("\n================================================================")
    print("STEP 3: Abstract Argumentation Semantics Resolution Matrices")
    print("================================================================")
    print(f"Total Conflict-Free Subset Configurations : {len(cf_sets)}")
    print(f"Total Admissible Self-Defending Sets      : {[list(s) for s in admissible_sets]}")
    print(f"Calculated Grounded Extension Set         : {list(grounded)}")
    print(f"Calculated Preferred Extensions Sets      : {[list(p) for p in preferred]}")

    print("\n================================================================")
    print("FINAL STEP: Contextualized Clinical Recommendations Output")
    print("================================================================")
    
    print(f"Target Health Objective: {target_health_goal}\n")
    
    viable_treatments = set()
    for ext in preferred:
        for arg_id in ext:
            arg_obj = engine.arguments[arg_id]
            if arg_obj.scheme_type == "ASPT":
                viable_treatments.add(arg_obj.metadata["treatment"])

    for rx in candidate_treatments:
        concept = patient_context.umls_mappings.get(rx)
        display_name = f"{concept.preferred_term} ({rx})" if concept else rx
        
        if rx in viable_treatments:
            print(f" ACCEPTED OPTION: {display_name}")
            print(f"   Status: Successfully passed all safety critical questions (CQs 1-4). Safe to deploy.")
        else:
            print(f" REJECTED OPTION: {display_name}")
            aspt_id = [k for k, v in engine.arguments.items() if v.scheme_type == "ASPT" and v.metadata["treatment"] == rx][0]
            attackers = aaf_solver.get_attackers(aspt_id)
            
            for attacker_id in attackers:
                attacker_arg = engine.arguments[attacker_id]
                if attacker_arg.scheme_type != "ASPT":  
                    print(f"   Reason Trace -> [{attacker_arg.scheme_type}]: {attacker_arg.conclusion}")
        print("-" * 64)
