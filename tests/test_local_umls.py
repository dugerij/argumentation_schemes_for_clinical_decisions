import json
import sqlite3
from pathlib import Path

from clinical_cds.direct import load_direct_dataset
from clinical_cds.evaluation import exact_label_match
from clinical_cds.medqa import load_medqa_cases
from clinical_cds.normalization import UMLSNormalizer
from clinical_cds.terminology.local_umls import (
    LocalUMLSBuildConfig,
    LocalUMLSClient,
    build_local_umls_database,
    normalize_umls_term,
)


def _write_rrf(path: Path, rows: list[list[str]]) -> None:
    path.write_text("".join("|".join(row) + "|\n" for row in rows), encoding="utf-8")


def test_build_local_umls_database_and_query(tmp_path):
    meta_dir = tmp_path / "META"
    meta_dir.mkdir()
    _write_rrf(
        meta_dir / "MRSTY.RRF",
        [
            ["C0020538", "T047", "A1.1.1", "Disease or Syndrome", "AT1", ""],
        ],
    )
    _write_rrf(
        meta_dir / "MRCONSO.RRF",
        [
            ["C0020538", "ENG", "P", "L1", "PF", "S1", "Y", "A1", "", "", "", "SNOMEDCT_US", "PT", "38341003", "Hypertensive disorder", "0", "N", ""],
            ["C0020538", "ENG", "P", "L2", "PF", "S2", "N", "A2", "", "", "", "ICD10CM", "PT", "I10", "Hypertension", "0", "N", ""],
            ["C0020538", "ENG", "P", "L3", "PF", "S3", "N", "A3", "", "", "", "SNOMEDCT_US", "SY", "38341003", "High blood pressure", "0", "N", ""],
        ],
    )

    db_path = tmp_path / "umls.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=meta_dir,
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US", "ICD10CM"),
            batch_size=2,
        )
    )

    client = LocalUMLSClient(
        db_path=db_path,
        source_vocabularies=("SNOMEDCT_US", "ICD10CM"),
    )
    hypertension = client.best_match("hypertension")

    assert hypertension is not None
    assert hypertension.cui == "C0020538"
    assert hypertension.category == "diagnosis"
    assert client.supports_full_alias_lookup is True
    assert set(client.concept_terms(hypertension.cui)) == {
        "High blood pressure",
        "Hypertension",
        "Hypertensive disorder",
    }


def test_umls_normalizer_matches_diagnosis_synonyms(tmp_path):
    db_path = tmp_path / "umls.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=_seed_meta_dir(tmp_path),
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US",),
        )
    )

    normalizer = UMLSNormalizer(
        LocalUMLSClient(
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US",),
        )
    )

    assert (
        normalizer.diagnosis_key("hypertension")
        == normalizer.diagnosis_key("hypertensive disorder")
    )
    assert normalizer.match_diagnosis(
        "hypertension",
        ("Asthma", "Hypertensive disorder"),
    ) == "Hypertensive disorder"
    assert exact_label_match(
        "Hypertension",
        "Hypertensive disorder",
        normalizer,
    ) == 1.0
    assert "Hypertensive disorder" in normalizer.expand_text(
        "Persistent high blood pressure"
    )


def _seed_meta_dir(tmp_path: Path) -> Path:
    meta_dir = tmp_path / "META"
    meta_dir.mkdir(exist_ok=True)
    _write_rrf(
        meta_dir / "MRSTY.RRF",
        [["C0020538", "T047", "A1.1.1", "Disease or Syndrome", "AT1", ""]],
    )
    _write_rrf(
        meta_dir / "MRCONSO.RRF",
        [
            ["C0020538", "ENG", "P", "L1", "PF", "S1", "Y", "A1", "", "", "", "SNOMEDCT_US", "PT", "38341003", "Hypertensive disorder", "0", "N", ""],
            ["C0020538", "ENG", "P", "L2", "PF", "S2", "N", "A2", "", "", "", "SNOMEDCT_US", "SY", "38341003", "Hypertension", "0", "N", ""],
            ["C0020538", "ENG", "P", "L3", "PF", "S3", "N", "A3", "", "", "", "SNOMEDCT_US", "SY", "38341003", "High blood pressure", "0", "N", ""],
            ["C0020538", "ENG", "P", "L4", "PF", "S4", "Y", "A4", "", "", "", "ICD10CM", "PT", "I10", "Hypertension", "0", "N", ""],
        ],
    )
    return meta_dir


def test_normalize_umls_term_collapses_punctuation():
    assert normalize_umls_term("Stage-2 Hypertension / Acute") == "stage 2 hypertension acute"


def test_preferred_term_fallback_avoids_unindexed_alias_scan(tmp_path):
    db_path = tmp_path / "umls.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=_seed_meta_dir(tmp_path),
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US", "ICD10CM"),
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_umls_terms_cui_source")
        conn.commit()

    client = LocalUMLSClient(
        db_path=db_path,
        source_vocabularies=("SNOMEDCT_US", "ICD10CM"),
    )

    assert client.supports_full_alias_lookup is False
    assert set(client.concept_terms("C0020538")) == {
        "Hypertension",
        "Hypertensive disorder",
    }


def test_local_umls_persistent_lookup_cache_reuses_prior_result(
    tmp_path,
    monkeypatch,
):
    meta_dir = _seed_meta_dir(tmp_path)
    db_path = tmp_path / "umls.sqlite3"
    lookup_cache_path = tmp_path / "umls_lookup_cache.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=meta_dir,
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US",),
        )
    )

    first_client = LocalUMLSClient(
        db_path=db_path,
        source_vocabularies=("SNOMEDCT_US",),
        lookup_cache_db_path=lookup_cache_path,
    )
    first_match = first_client.best_match("hypertensive disorder")
    first_client.flush_cache()

    with sqlite3.connect(lookup_cache_path) as conn:
        rows = conn.execute(
            "SELECT normalized_term, source_vocabularies FROM umls_lookup_cache"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "hypertensive disorder"
    assert rows[0][1].endswith("|SNOMEDCT_US")

    second_client = LocalUMLSClient(
        db_path=db_path,
        source_vocabularies=("SNOMEDCT_US",),
        lookup_cache_db_path=lookup_cache_path,
    )
    monkeypatch.setattr(
        second_client,
        "search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Persistent cache miss")
        ),
    )
    second_match = second_client.best_match("hypertensive disorder")

    assert first_match is not None
    assert second_match is not None
    assert second_match.cui == first_match.cui


def test_medqa_graph_coverage_uses_umls_equivalence(tmp_path, direct_root):
    db_path = tmp_path / "umls.sqlite3"
    build_local_umls_database(
        LocalUMLSBuildConfig(
            meta_dir=_seed_meta_dir(tmp_path),
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US",),
        )
    )
    normalizer = UMLSNormalizer(
        LocalUMLSClient(
            db_path=db_path,
            source_vocabularies=("SNOMEDCT_US",),
        )
    )
    medqa_dir = tmp_path / "medqa"
    medqa_dir.mkdir()
    question = {
        "question": "Which of the following is the most likely diagnosis?",
        "answer": "Hypertensive disorder",
        "options": {
            "A": "Asthma",
            "B": "Hypertensive disorder",
        },
    }
    (medqa_dir / "test.jsonl").write_text(
        json.dumps(question) + "\n",
        encoding="utf-8",
    )

    cases = load_medqa_cases(
        medqa_dir,
        graphs=load_direct_dataset(direct_root).graphs,
        normalizer=normalizer,
    )

    assert len(cases) == 1
    assert cases[0].metadata["direct_graph_covered"] is True
    assert cases[0].disease_category == "Hypertension"
