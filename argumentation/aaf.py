from itertools import combinations

from argumentation.schemes import Argument


class AbstractArgumentationFramework:
    def __init__(self, arguments: set[Argument], attacks: set[tuple[str, str]]):
        self.arguments = arguments
        self.arg_ids = {arg.id for arg in arguments}
        self.attacks = attacks

    def get_attackers(self, arg_id: str) -> set[str]:
        return {attacker for (attacker, attacked) in self.attacks if attacked == arg_id}

    def is_conflict_free(self, subset: set[str]) -> bool:
        return not any((a, b) in self.attacks for a in subset for b in subset)

    def d_operator(self, subset: set[str]) -> set[str]:
        acceptable = set()
        for arg in self.arg_ids:
            attackers = self.get_attackers(arg)
            if all(any((defender, attacker) in self.attacks for defender in subset) for attacker in attackers):
                acceptable.add(arg)
        return acceptable

    def compute_all_conflict_free_sets(self) -> list[set[str]]:
        cf_sets = []
        list_ids = list(self.arg_ids)
        for size in range(len(list_ids) + 1):
            for combination in combinations(list_ids, size):
                subset = set(combination)
                if self.is_conflict_free(subset):
                    cf_sets.append(subset)
        return cf_sets

    def compute_all_admissible_sets(self) -> list[set[str]]:
        admissible_sets = []
        for subset in self.compute_all_conflict_free_sets():
            if subset.issubset(self.d_operator(subset)):
                admissible_sets.append(subset)
        return admissible_sets

    def compute_grounded_extension(self) -> set[str]:
        current_set = set()
        while True:
            next_set = self.d_operator(current_set)
            if next_set == current_set:
                return current_set
            current_set = next_set

    def compute_preferred_extensions(self) -> list[set[str]]:
        admissible_sets = self.compute_all_admissible_sets()
        preferred = []
        for candidate in admissible_sets:
            if not any(candidate != other and candidate.issubset(other) for other in admissible_sets):
                preferred.append(candidate)
        return preferred
