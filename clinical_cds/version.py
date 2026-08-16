"""Identity of the supported clinical argumentation method."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MethodRelease:
    release_id: str
    dialogue_id: str
    display_name: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


CURRENT_RELEASE = MethodRelease(
    release_id="clinical-argumentation-version-iii",
    dialogue_id="canonical-clinical-argumentation-version-iii-direct-differential-rag",
    display_name="Clinical Argumentation Version III",
)
