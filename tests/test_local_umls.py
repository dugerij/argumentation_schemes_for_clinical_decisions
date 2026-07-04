from pathlib import Path

from retrieval.concepts.local_umls import (
    LocalUMLSBuildConfig,
    LocalUMLSClient,
    build_local_umls_database,
    normalize_umls_term,
)
from retrieval.concepts.umls import UMLSConfig, create_umls_client


def _write_rrf(path: Path, rows: list[list[str]]) -> None:
    path.write_text("".join("|".join(row) + "|\n" for row in rows), encoding="utf-8")


def test_build_local_umls_database_and_query(tmp_path):
    meta_dir = tmp_path / "META"
    meta_dir.mkdir()
    _write_rrf(
        meta_dir / "MRSTY.RRF",
        [
            ["C0020538", "T047", "A1.1.1", "Disease or Syndrome", "AT1", ""],
            ["C0004057", "T121", "A1.2.3", "Pharmacologic Substance", "AT2", ""],
        ],
    )
    _write_rrf(
        meta_dir / "MRCONSO.RRF",
        [
            ["C0020538", "ENG", "P", "L1", "PF", "S1", "Y", "A1", "", "", "", "SNOMEDCT_US", "PT", "38341003", "Hypertensive disorder", "0", "N", ""],
            ["C0020538", "ENG", "P", "L2", "PF", "S2", "N", "A2", "", "", "", "ICD10CM", "PT", "I10", "Hypertension", "0", "N", ""],
            ["C0004057", "ENG", "P", "L3", "PF", "S3", "Y", "A3", "", "", "", "RXNORM", "PT", "17767", "Amlodipine", "0", "N", ""],
        ],
    )

    db_path = tmp_path / "umls.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=meta_dir,
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US", "ICD10CM", "RXNORM"),
            batch_size=2,
        )
    )

    client = LocalUMLSClient(db_path=db_path, source_vocabularies=("SNOMEDCT_US", "ICD10CM", "RXNORM"))
    hypertension = client.best_match("hypertension")
    amlodipine = client.best_match("amlodipine")

    assert hypertension is not None
    assert hypertension.cui == "C0020538"
    assert hypertension.category == "diagnosis"
    assert amlodipine is not None
    assert amlodipine.source_vocabulary == "RXNORM"
    assert amlodipine.category == "medication"


def test_create_umls_client_uses_local_backend(tmp_path):
    db_path = tmp_path / "umls.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=_seed_meta_dir(tmp_path),
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US",),
        )
    )

    client = create_umls_client(
        UMLSConfig(
            backend="local",
            local_db_path=str(db_path),
            source_vocabularies=("SNOMEDCT_US",),
        )
    )

    match = client.best_match("hypertensive disorder")
    assert match is not None
    assert match.cui == "C0020538"


def _seed_meta_dir(tmp_path: Path) -> Path:
    meta_dir = tmp_path / "META"
    meta_dir.mkdir(exist_ok=True)
    _write_rrf(
        meta_dir / "MRSTY.RRF",
        [["C0020538", "T047", "A1.1.1", "Disease or Syndrome", "AT1", ""]],
    )
    _write_rrf(
        meta_dir / "MRCONSO.RRF",
        [["C0020538", "ENG", "P", "L1", "PF", "S1", "Y", "A1", "", "", "", "SNOMEDCT_US", "PT", "38341003", "Hypertensive disorder", "0", "N", ""]],
    )
    return meta_dir


def test_normalize_umls_term_collapses_punctuation():
    assert normalize_umls_term("Stage-2 Hypertension / Acute") == "stage 2 hypertension acute"
