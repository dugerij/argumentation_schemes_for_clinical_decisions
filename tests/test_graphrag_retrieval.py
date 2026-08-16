from clinical_cds.schema import ClinicalCase
from types import SimpleNamespace
from graphrag_runtime.corpus import ControlledPremise
from graphrag_runtime.retrieval import (
    DensePremiseIndex,
    DensePremiseNeighbor,
    FixedGraphRagKnowledgeRetriever,
    flat_seeded_graph_candidate_choices,
    IndependentFlatPremiseRetriever,
    select_diverse_candidate_routes,
    rank_specific_candidate_choices,
    _numeric_threshold_match,
    _positive_diagnostic_event_match,
    _structured_evidence_strength,
)
from graphrag_runtime.provenance_contract import canonical_sha256


SOURCE_ONE = "source-chunk:" + "1" * 20


class _EventConceptNormalizer:
    _concepts = {
        "segmental filling defects": "C:event:filling-defect",
        "filling defects": "C:event:filling-defect",
        "mucosal erosions": "C:event:mucosal-erosion",
        "erosions": "C:event:mucosal-erosion",
        "intracellular inclusions": "C:event:inclusion",
        "inclusions": "C:event:inclusion",
        "focal extra axial collection": "C:event:collection",
        "extra axial collection": "C:event:collection",
        "diffuse cortical atrophy": "C:event:atrophy",
        "cortical atrophy": "C:event:atrophy",
        "invasive abnormal tissue": "C:event:abnormal-tissue",
        "abnormal tissue": "C:event:abnormal-tissue",
        "blood clot": "C:event:thrombus",
        "thrombus": "C:event:thrombus",
    }

    def concept(self, term):
        cui = self._concepts.get(" ".join(term.casefold().replace("-", " ").split()))
        return SimpleNamespace(cui=cui) if cui else None


EVENT_NORMALIZER = _EventConceptNormalizer()


def _premise(source: str, graph: str, label: str) -> ControlledPremise:
    return ControlledPremise(
        id=source,
        title=f"{graph} | {label}",
        text=f"Evidence for {label}.",
        graph_id=graph,
        category=graph,
        node_id=f"premise:{source[-4:]}",
        diagnosis_label=label,
        premise_type="Diagnostic Criteria",
        diagnostic_path=(f"Suspected {graph}", label),
        source_chunk_id=source,
        knowledge_source_ids=(f"guideline:{graph}",),
        source_origin="controlled_direct_graph",
    )


def test_multi_route_selection_is_diverse_specific_and_budgeted():
    sources = [f"source-chunk:{index:020x}" for index in range(1, 15)]
    corpus = (
        _premise(sources[0], "Cardiac", "Suspected Cardiac"),
        _premise(sources[1], "Cardiac", "NSTEMI"),
        _premise(sources[2], "Cardiac", "NSTEMI"),
        _premise(sources[3], "Pulmonary", "Pulmonary Embolism"),
        _premise(sources[4], "Pulmonary", "Pulmonary Embolism"),
        _premise(sources[5], "Neurology", "Ischaemic Stroke"),
        _premise(sources[6], "Neurology", "Ischaemic Stroke"),
        _premise(sources[7], "Digestive", "Peptic Ulcer Disease"),
        _premise(sources[8], "Digestive", "Peptic Ulcer Disease"),
        *(
            _premise(source, "Cardiac", "NSTEMI")
            for source in sources[9:]
        ),
    )
    choices = (
        ("candidate:generic", "Suspected Cardiac", (sources[0],)),
        ("candidate:nstemi", "NSTEMI", tuple(
            source for source in sources if source in {
                sources[1], sources[2], *sources[9:]
            }
        )),
        ("candidate:pe", "Pulmonary Embolism", (sources[3], sources[4])),
        ("candidate:stroke", "Ischaemic Stroke", (sources[5], sources[6])),
        ("candidate:pud", "Peptic Ulcer Disease", (sources[7], sources[8])),
    )

    selected = select_diverse_candidate_routes(
        choices, corpus, maximum_routes=4
    )

    assert [row["diagnosis_label"] for row in selected] == [
        "NSTEMI",
        "Pulmonary Embolism",
        "Ischaemic Stroke",
        "Peptic Ulcer Disease",
    ]
    assert sum(len(row["source_chunk_ids"]) for row in selected) == 12
    assert len({
        source
        for row in selected
        for source in row["source_chunk_ids"]
    }) == 12


def test_candidate_ranking_removes_generic_and_meaningless_routes():
    generic_source = "source-chunk:" + "a" * 20
    noise_source = "source-chunk:" + "b" * 20
    criterion_source = "source-chunk:" + "c" * 20
    risk_source = "source-chunk:" + "d" * 20
    generic = _premise(generic_source, "ACS", "Suspected ACS")
    noise = ControlledPremise(
        **{**_premise(noise_source, "ACS", "NSTEMI").__dict__, "text": "etc."}
    )
    criterion = _premise(criterion_source, "Pulmonary", "Pulmonary Embolism")
    risk = ControlledPremise(
        **{
            **_premise(risk_source, "Metabolic", "Type II Diabetes").__dict__,
            "premise_type": "Risk Factors",
        }
    )
    choices = (
        ("candidate:generic", "Suspected ACS", (generic_source,)),
        ("candidate:noise", "NSTEMI", (noise_source,)),
        ("candidate:risk", "Type II Diabetes", (risk_source,)),
        ("candidate:criterion", "Pulmonary Embolism", (criterion_source,)),
    )

    ranked = rank_specific_candidate_choices(
        choices,
        (generic, noise, criterion, risk),
        query_text="Pulmonary embolism with diagnostic evidence",
    )

    assert [label for _, label, _ in ranked] == [
        "Pulmonary Embolism",
        "Type II Diabetes",
    ]


def test_length_normalization_suppresses_verbose_incidental_overlap():
    concise_source = "source-chunk:" + "e" * 20
    verbose_source = "source-chunk:" + "f" * 20
    concise = ControlledPremise(
        **{
            **_premise(concise_source, "Endocrine", "Hyperthyroidism").__dict__,
            "text": "Suppressed TSH with elevated free T4.",
        }
    )
    verbose = ControlledPremise(
        **{
            **_premise(verbose_source, "Pulmonary", "Submassive PE").__dict__,
            "text": (
                "The patient is stable with possible functional impairment and "
                "many general clinical findings requiring hospitalization and "
                "monitoring. Biomarkers may be elevated in the patient."
            ),
        }
    )
    choices = (
        ("candidate:pe", "Submassive PE", (verbose_source,)),
        ("candidate:thyroid", "Hyperthyroidism", (concise_source,)),
    )

    ranked = rank_specific_candidate_choices(
        choices,
        (concise, verbose),
        query_text="TSH suppressed and free T4 elevated",
    )

    assert ranked[0][1] == "Hyperthyroidism"


def test_flat_seed_expands_to_other_specific_diagnoses_in_same_graph():
    symptom_source = "source-chunk:" + "7" * 20
    subtype_source = "source-chunk:" + "8" * 20
    distractor_source = "source-chunk:" + "9" * 20
    symptom = ControlledPremise(
        **{
            **_premise(symptom_source, "Thyroid", "Hyperthyroidism").__dict__,
            "text": "Palpitations tremor heat intolerance and weight loss.",
        }
    )
    subtype = ControlledPremise(
        **{
            **_premise(subtype_source, "Thyroid", "Graves Disease").__dict__,
            "text": "TSH receptor antibodies support Graves disease.",
        }
    )
    distractor = ControlledPremise(
        **{
            **_premise(distractor_source, "Renal", "Nephrotic Syndrome").__dict__,
            "text": "Heavy proteinuria and edema.",
        }
    )

    choices = flat_seeded_graph_candidate_choices(
        "Palpitations tremor and weight loss.",
        (symptom, subtype, distractor),
        maximum_seed_graphs=1,
    )

    assert [label for _, label, _ in choices] == [
        "Hyperthyroidism",
        "Graves Disease",
    ]
    assert choices[1][2] == (subtype_source,)


def test_flat_seeded_graph_candidates_exclude_generic_and_noise_nodes():
    useful_source = "source-chunk:" + "4" * 20
    generic_source = "source-chunk:" + "5" * 20
    noise_source = "source-chunk:" + "6" * 20
    useful = _premise(useful_source, "Cardiac", "NSTEMI")
    generic = _premise(generic_source, "Cardiac", "Suspected ACS")
    noise = ControlledPremise(
        **{**_premise(noise_source, "Cardiac", "Other ACS").__dict__, "text": "etc."}
    )

    choices = flat_seeded_graph_candidate_choices(
        "Evidence for NSTEMI.",
        (useful, generic, noise),
    )

    assert [label for _, label, _ in choices] == ["NSTEMI"]


def test_dense_neighbor_recovers_route_with_no_lexical_query_overlap():
    relevant_source = "source-chunk:" + "a1" * 10
    lexical_source = "source-chunk:" + "b2" * 10
    relevant = ControlledPremise(
        **{
            **_premise(relevant_source, "Cardiac", "Acute Myocardial Infarction").__dict__,
            "text": "Acute coronary artery occlusion with myocardial necrosis.",
        }
    )
    lexical = ControlledPremise(
        **{
            **_premise(lexical_source, "Musculoskeletal", "Costochondritis").__dict__,
            "text": "Chest wall tenderness and reproducible pain.",
        }
    )

    choices = flat_seeded_graph_candidate_choices(
        "Heart attack with raised cardiac enzymes.",
        (relevant, lexical),
        dense_neighbors=(DensePremiseNeighbor(relevant_source, 0.91, 1),),
        maximum_seed_graphs=2,
    )

    assert choices[0][1] == "Acute Myocardial Infarction"
    assert choices[0][2] == (relevant_source,)


def test_dense_premise_index_max_pools_independent_patient_findings():
    first_source = "source-chunk:" + "c3" * 10
    second_source = "source-chunk:" + "d4" * 10
    corpus = (
        _premise(first_source, "Cardiac", "NSTE-ACS"),
        _premise(second_source, "Pulmonary", "Pulmonary Embolism"),
    )
    index = DensePremiseIndex(corpus, ((1.0, 0.0), (0.0, 1.0)))

    neighbors = index.nearest(((0.95, 0.05), (0.1, 0.9)), maximum_neighbors=2)

    assert {item.source_chunk_id for item in neighbors} == {
        first_source,
        second_source,
    }
    assert all(item.score > 0.89 for item in neighbors)


def test_hybrid_retrieval_still_emits_exactly_eight_routes():
    sources = tuple(f"source-chunk:{index:020x}" for index in range(20, 30))
    corpus = tuple(
        _premise(source, f"Graph-{index}", f"Diagnosis-{index}")
        for index, source in enumerate(sources)
    )
    choices = flat_seeded_graph_candidate_choices(
        "No lexical overlap is required for this dense ranking.",
        corpus,
        dense_neighbors=tuple(
            DensePremiseNeighbor(source, 0.99 - index / 100.0, index + 1)
            for index, source in enumerate(sources)
        ),
    )

    selected = select_diverse_candidate_routes(choices, corpus)

    assert len(choices) == 10
    assert len(selected) == 8


def test_multiple_independent_premises_outrank_one_dense_neighbor():
    single_source = "source-chunk:" + "e5" * 10
    first_support = "source-chunk:" + "f6" * 10
    second_support = "source-chunk:" + "a7" * 10
    corpus = (
        _premise(single_source, "Single", "Single-Match Diagnosis"),
        _premise(first_support, "Covered", "Multi-Evidence Diagnosis"),
        _premise(second_support, "Covered", "Multi-Evidence Diagnosis"),
    )

    choices = flat_seeded_graph_candidate_choices(
        "A semantic-only patient query.",
        corpus,
        dense_neighbors=(
            DensePremiseNeighbor(single_source, 0.95, 1),
            DensePremiseNeighbor(first_support, 0.90, 2),
            DensePremiseNeighbor(second_support, 0.88, 3),
        ),
    )

    assert choices[0][1] == "Multi-Evidence Diagnosis"


def test_family_first_selector_defers_duplicate_graph_subtype():
    sources = tuple(f"source-chunk:{index:020x}" for index in range(40, 43))
    corpus = (
        _premise(sources[0], "ACS", "NSTEMI"),
        _premise(sources[1], "ACS", "Unstable Angina"),
        _premise(sources[2], "Pulmonary", "Pulmonary Embolism"),
    )
    choices = (
        ("candidate:nstemi", "NSTEMI", (sources[0],)),
        ("candidate:ua", "Unstable Angina", (sources[1],)),
        ("candidate:pe", "Pulmonary Embolism", (sources[2],)),
    )

    selected = select_diverse_candidate_routes(
        choices,
        corpus,
        maximum_routes=2,
        maximum_facts=2,
    )

    assert [item["diagnosis_label"] for item in selected] == [
        "NSTEMI",
        "Pulmonary Embolism",
    ]
    assert [item["family_assignment_method"] for item in selected] == [
        "graph_membership",
        "graph_membership",
    ]
    assert [item["original_candidate_rank"] for item in selected] == [1, 3]


def test_family_first_selector_refills_with_siblings_after_unique_graphs():
    sources = tuple(f"source-chunk:{index:020x}" for index in range(43, 46))
    corpus = (
        _premise(sources[0], "ACS", "NSTEMI"),
        _premise(sources[1], "ACS", "Unstable Angina"),
        _premise(sources[2], "Pulmonary", "Pulmonary Embolism"),
    )
    choices = (
        ("candidate:nstemi", "NSTEMI", (sources[0],)),
        ("candidate:ua", "Unstable Angina", (sources[1],)),
        ("candidate:pe", "Pulmonary Embolism", (sources[2],)),
    )

    selected = select_diverse_candidate_routes(
        choices,
        corpus,
        maximum_routes=3,
        maximum_facts=3,
    )

    assert [item["diagnosis_label"] for item in selected] == [
        "NSTEMI",
        "Pulmonary Embolism",
        "Unstable Angina",
    ]
    assert [
        alternative["diagnosis_label"]
        for alternative in selected[0]["family_alternatives"]
    ] == ["NSTEMI", "Unstable Angina"]
    assert selected[0]["family_alternatives"][0]["representative"] is True
    assert selected[0]["family_alternatives"][1]["representative"] is False


def test_sibling_ranking_prefers_route_with_distinct_multi_evidence():
    sources = tuple(f"source-chunk:{index:020x}" for index in range(60, 63))
    corpus = (
        _premise(sources[0], "ACS", "NSTEMI"),
        _premise(sources[1], "ACS", "NSTEMI"),
        _premise(sources[2], "ACS", "Unstable Angina"),
    )

    choices = flat_seeded_graph_candidate_choices(
        "Semantic evidence without shared lexical terms.",
        corpus,
        dense_neighbors=(
            DensePremiseNeighbor(sources[2], 0.95, 1),
            DensePremiseNeighbor(sources[0], 0.92, 2),
            DensePremiseNeighbor(sources[1], 0.90, 3),
        ),
    )

    assert choices[0][1] == "NSTEMI"


def test_extra_fact_budget_prioritizes_selected_sibling_routes():
    sources = tuple(f"source-chunk:{index:020x}" for index in range(70, 75))
    corpus = (
        _premise(sources[0], "ACS", "NSTEMI"),
        _premise(sources[1], "ACS", "NSTEMI"),
        _premise(sources[2], "ACS", "Unstable Angina"),
        _premise(sources[3], "ACS", "Unstable Angina"),
        _premise(sources[4], "Pulmonary", "Pulmonary Embolism"),
    )
    choices = (
        ("candidate:nstemi", "NSTEMI", (sources[0], sources[1])),
        ("candidate:ua", "Unstable Angina", (sources[2], sources[3])),
        ("candidate:pe", "Pulmonary Embolism", (sources[4],)),
    )

    selected = select_diverse_candidate_routes(
        choices,
        corpus,
        maximum_routes=3,
        maximum_facts=5,
    )

    assert [item["diagnosis_label"] for item in selected] == [
        "NSTEMI",
        "Pulmonary Embolism",
        "Unstable Angina",
    ]
    assert [len(item["source_chunk_ids"]) for item in selected] == [2, 1, 2]


def test_atomic_premise_passages_collapse_back_to_one_provenance_source():
    first_source = "source-chunk:" + "b8" * 10
    second_source = "source-chunk:" + "c9" * 10
    corpus = (
        _premise(first_source, "Cardiac", "HFrEF"),
        _premise(second_source, "Pulmonary", "Pulmonary Embolism"),
    )
    index = DensePremiseIndex(
        corpus,
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        source_indices=(0, 0, 1),
    )

    neighbors = index.nearest(((0.0, 1.0),), maximum_neighbors=2)

    assert neighbors[0].source_chunk_id == first_source
    assert len({item.source_chunk_id for item in neighbors}) == len(neighbors)


def test_numeric_threshold_matching_executes_named_clinical_measurements():
    assert _numeric_threshold_match(
        "Echocardiogram shows LVEF=30%.",
        "HFrEF diagnostic criterion: LVEF≤40%.",
    )
    assert _numeric_threshold_match(
        "Physical examination BP: 200/106.",
        "SBP≥140mmHg or DBP≥90mmHg confirms hypertension.",
    )
    assert not _numeric_threshold_match(
        "Echocardiogram shows LVEF=55%.",
        "HFrEF diagnostic criterion: LVEF≤40%.",
    )


def test_structured_measurement_cannot_be_dropped_by_dense_family_budget():
    sources = tuple(f"source-chunk:{index:020x}" for index in range(100, 113))
    routine = tuple(
        ControlledPremise(
            **{
                **_premise(source, f"Graph-{index}", f"Diagnosis-{index}").__dict__,
                "text": f"Dense semantic premise number {index}.",
            }
        )
        for index, source in enumerate(sources[:-1])
    )
    threshold = ControlledPremise(
        **{
            **_premise(sources[-1], "Graph-structured", "Threshold condition").__dict__,
            "text": "A named criterion is SBP≥140 mmHg or DBP≥90 mmHg.",
        }
    )
    corpus = (*routine, threshold)
    choices = flat_seeded_graph_candidate_choices(
        "Current examination BP: 200/106.",
        corpus,
        dense_neighbors=tuple(
            DensePremiseNeighbor(source, 0.99 - index / 100.0, index + 1)
            for index, source in enumerate(sources[:-1])
        ),
    )
    selected = select_diverse_candidate_routes(
        choices, corpus, query_text="Current examination BP: 200/106."
    )

    assert choices[0][1] == "Threshold condition"
    assert "Threshold condition" in {
        route["diagnosis_label"] for route in selected
    }


def test_historical_diagnosis_does_not_receive_current_event_boost():
    premise = "Biopsy demonstrates invasive abnormal tissue."

    assert _structured_evidence_strength(
        "History of biopsy demonstrating invasive abnormal tissue.", premise,
        normalizer=EVENT_NORMALIZER,
    ) == 0.0
    assert _structured_evidence_strength(
        "Current biopsy demonstrates invasive abnormal tissue.", premise,
        normalizer=EVENT_NORMALIZER,
    ) == 0.8


def test_positive_event_requires_exact_content_not_related_modalities():
    assert _positive_diagnostic_event_match(
        "Current chest CT demonstrates segmental filling defects.",
        "Chest CT demonstrates segmental filling defects.",
        normalizer=EVENT_NORMALIZER,
    )
    assert _positive_diagnostic_event_match(
        "Current CT shows a blood clot.",
        "Computed tomography detects thrombus.",
        normalizer=EVENT_NORMALIZER,
    )
    assert not _positive_diagnostic_event_match(
        "Current CT shows a blood clot.",
        "Computed tomography detects thrombus.",
    )
    assert not _positive_diagnostic_event_match(
        "Current coronary angiography shows unobstructed arteries.",
        "Endoscopy demonstrates mucosal erosions.",
        normalizer=EVENT_NORMALIZER,
    )
    assert not _positive_diagnostic_event_match(
        "Current cranial CT scan shows a focal extra-axial collection.",
        "A CT scan demonstrates diffuse cortical atrophy.",
        normalizer=EVENT_NORMALIZER,
    )
    assert _positive_diagnostic_event_match(
        "Current microscopy shows intracellular inclusions.",
        "Microscopy detects intracellular inclusions.",
        normalizer=EVENT_NORMALIZER,
    )


def test_family_seed_expands_structured_child_to_its_parent_graph():
    criterion_source = "source-chunk:" + "d1" * 10
    sibling_source = "source-chunk:" + "e2" * 10
    criterion = ControlledPremise(
        **{
            **_premise(criterion_source, "Parent family", "Measured subtype").__dict__,
            "text": "Measured subtype criterion: index≤40.",
        }
    )
    sibling = ControlledPremise(
        **{
            **_premise(sibling_source, "Parent family", "Sibling subtype").__dict__,
            "text": "A separate sibling criterion.",
        }
    )

    choices = flat_seeded_graph_candidate_choices(
        "Current index=30.", (criterion, sibling), maximum_seed_graphs=1
    )

    assert [label for _, label, _ in choices] == [
        "Measured subtype",
        "Sibling subtype",
    ]


def test_audited_graphrag_output_becomes_provenance_complete_fixed_bundle():
    premise = ControlledPremise(
        id=SOURCE_ONE,
        title="Digestive | Gastritis | Criteria",
        text="Endoscopic inflammation supports gastritis.",
        graph_id="direct:gastritis",
        category="Digestive",
        node_id="premise:gastritis:1",
        diagnosis_label="Gastritis",
        premise_type="Diagnostic Criteria",
        diagnostic_path=("Digestive", "Gastritis"),
        source_chunk_id=SOURCE_ONE,
        knowledge_source_ids=("guideline:1",),
        source_origin="provenance_backed_project_extension",
    )
    retriever = FixedGraphRagKnowledgeRetriever(
        (premise,),
        ({
            "case_id": "case-1",
            "ranked_candidates": [{
                "rank": 1,
                "diagnosis_label": "Gastritis",
                "kg_bindings": [{"source_chunk_id": SOURCE_ONE}],
            }],
            "citation_allowlist": [SOURCE_ONE],
            "citation_allowlist_sha256": canonical_sha256([SOURCE_ONE]),
            "error": None,
        },),
    )
    case = ClinicalCase(
        case_id="case-1",
        dataset="direct",
        task="diagnosis",
        sections={"history_of_present_illness": "Epigastric pain."},
        gold_label="Gastritis",
    )

    bundle = retriever.retrieve(case, top_k=8)

    assert bundle.facts[0].evidence_id == "K1"
    assert bundle.facts[0].node_id == premise.node_id
    assert bundle.facts[0].diagnostic_path == premise.diagnostic_path
    assert bundle.facts[0].source_chunk_id == premise.source_chunk_id
    assert bundle.facts[0].knowledge_source_ids == premise.knowledge_source_ids
    assert bundle.citation_allowlist == (SOURCE_ONE,)


def test_fixed_bundle_adds_sibling_evidence_without_an_extra_family_slot():
    primary_source = "source-chunk:" + "ab" * 10
    sibling_source = "source-chunk:" + "cd" * 10
    pulmonary_source = "source-chunk:" + "ef" * 10
    corpus = (
        _premise(primary_source, "ACS", "NSTEMI"),
        _premise(sibling_source, "ACS", "Unstable Angina"),
        _premise(pulmonary_source, "Pulmonary", "Pulmonary Embolism"),
    )
    output = {
        "case_id": "case-family",
        "ranked_candidates": [
            {
                "rank": 1,
                "diagnosis_label": "NSTEMI",
                "kg_bindings": [{"source_chunk_id": primary_source}],
            },
            {
                "rank": 2,
                "diagnosis_label": "Pulmonary Embolism",
                "kg_bindings": [{"source_chunk_id": pulmonary_source}],
            },
        ],
        "family_routes": [
            {
                "family_rank": 1,
                "graph_id": "ACS",
                "family_key": "graph:ACS",
                "representative_diagnosis": "NSTEMI",
                "alternatives": [
                    {
                        "candidate_id": "candidate:nstemi",
                        "diagnosis_label": "NSTEMI",
                        "graph_id": "ACS",
                        "diagnostic_path": ["Suspected ACS", "NSTEMI"],
                        "source_chunk_ids": [primary_source],
                        "original_candidate_rank": 1,
                        "representative": True,
                    },
                    {
                        "candidate_id": "candidate:ua",
                        "diagnosis_label": "Unstable Angina",
                        "graph_id": "ACS",
                        "diagnostic_path": ["Suspected ACS", "Unstable Angina"],
                        "source_chunk_ids": [sibling_source],
                        "original_candidate_rank": 3,
                        "representative": False,
                    },
                ],
            },
            {
                "family_rank": 2,
                "graph_id": "Pulmonary",
                "family_key": "graph:Pulmonary",
                "representative_diagnosis": "Pulmonary Embolism",
                "alternatives": [{
                    "candidate_id": "candidate:pe",
                    "diagnosis_label": "Pulmonary Embolism",
                    "graph_id": "Pulmonary",
                    "diagnostic_path": ["Suspected Pulmonary", "Pulmonary Embolism"],
                    "source_chunk_ids": [pulmonary_source],
                    "original_candidate_rank": 2,
                    "representative": True,
                }],
            },
        ],
        "citation_allowlist": [
            primary_source,
            sibling_source,
            pulmonary_source,
        ],
        "citation_allowlist_sha256": canonical_sha256([
            primary_source,
            sibling_source,
            pulmonary_source,
        ]),
        "error": None,
    }
    retriever = FixedGraphRagKnowledgeRetriever(corpus, (output,))
    case = ClinicalCase(
        case_id="case-family",
        dataset="direct",
        task="diagnosis",
        sections={"history": "Chest pain."},
        gold_label="Unstable Angina",
    )

    bundle = retriever.retrieve(case, top_k=3)

    assert len(bundle.family_routes) == 2
    assert [fact.diagnosis_label for fact in bundle.facts] == [
        "NSTEMI",
        "Pulmonary Embolism",
        "Unstable Angina",
    ]
    assert len(bundle.family_routes[0].alternatives) == 2
    assert {
        (item.graph_id, item.fact.diagnosis_label)
        for item in bundle.family_child_facts
    } == {
        ("ACS", "NSTEMI"),
        ("ACS", "Unstable Angina"),
        ("Pulmonary", "Pulmonary Embolism"),
    }


def test_audited_graphrag_output_accepts_exact_controlled_path_candidate():
    premise = ControlledPremise(
        id=SOURCE_ONE,
        title="Digestive | Gastritis | Criteria",
        text="Endoscopic inflammation supports gastritis.",
        graph_id="direct:gastritis",
        category="Digestive",
        node_id="premise:gastritis:1",
        diagnosis_label="Gastritis",
        premise_type="Diagnostic Criteria",
        diagnostic_path=("Digestive disorder", "Gastritis"),
        source_chunk_id=SOURCE_ONE,
        knowledge_source_ids=("guideline:1",),
        source_origin="provenance_backed_project_extension",
    )
    retriever = FixedGraphRagKnowledgeRetriever(
        (premise,),
        ({
            "case_id": "case-1",
            "ranked_candidates": [{
                "rank": 1,
                "diagnosis_label": "Digestive disorder",
                "kg_bindings": [{"source_chunk_id": SOURCE_ONE}],
            }],
            "citation_allowlist": [SOURCE_ONE],
            "citation_allowlist_sha256": canonical_sha256([SOURCE_ONE]),
            "error": None,
        },),
    )
    case = ClinicalCase(
        case_id="case-1",
        dataset="direct",
        task="diagnosis",
        sections={"history_of_present_illness": "Epigastric pain."},
        gold_label="Gastritis",
    )

    bundle = retriever.retrieve(case, top_k=8)

    assert bundle.facts[0].source_chunk_id == SOURCE_ONE
    assert bundle.facts[0].diagnosis_label == "Gastritis"


def test_audited_graphrag_output_rejects_label_absent_from_controlled_path():
    premise = ControlledPremise(
        id=SOURCE_ONE,
        title="Digestive | Gastritis | Criteria",
        text="Endoscopic inflammation supports gastritis.",
        graph_id="direct:gastritis",
        category="Digestive",
        node_id="premise:gastritis:1",
        diagnosis_label="Gastritis",
        premise_type="Diagnostic Criteria",
        diagnostic_path=("Digestive disorder", "Gastritis"),
        source_chunk_id=SOURCE_ONE,
        knowledge_source_ids=("guideline:1",),
        source_origin="provenance_backed_project_extension",
    )
    retriever = FixedGraphRagKnowledgeRetriever(
        (premise,),
        ({
            "case_id": "case-1",
            "ranked_candidates": [{
                "rank": 1,
                "diagnosis_label": "Cardiac disorder",
                "kg_bindings": [{"source_chunk_id": SOURCE_ONE}],
            }],
            "citation_allowlist": [SOURCE_ONE],
            "citation_allowlist_sha256": canonical_sha256([SOURCE_ONE]),
            "error": None,
        },),
    )
    case = ClinicalCase(
        case_id="case-1",
        dataset="direct",
        task="diagnosis",
        sections={"history_of_present_illness": "Epigastric pain."},
        gold_label="Gastritis",
    )

    try:
        retriever.retrieve(case, top_k=8)
    except ValueError as exc:
        assert str(exc) == "GraphRAG binding does not support its candidate."
    else:
        raise AssertionError("A diagnosis absent from the controlled path was accepted.")


def test_independent_flat_retriever_ranks_controlled_premises_without_graphrag():
    gastritis = ControlledPremise(
        id=SOURCE_ONE,
        title="Digestive | Gastritis | Criteria",
        text="Endoscopic inflammation and epigastric pain support gastritis.",
        graph_id="direct:gastritis",
        category="Digestive",
        node_id="premise:gastritis:1",
        diagnosis_label="Gastritis",
        premise_type="Diagnostic Criteria",
        diagnostic_path=("Digestive", "Gastritis"),
        source_chunk_id=SOURCE_ONE,
        knowledge_source_ids=("guideline:1",),
        source_origin="provenance_backed_project_extension",
    )
    other = ControlledPremise(
        id="source-chunk:" + "2" * 20,
        title="Cardiac | Arrhythmia | Criteria",
        text="Irregular tachycardia supports arrhythmia.",
        graph_id="direct:arrhythmia",
        category="Cardiac",
        node_id="premise:arrhythmia:1",
        diagnosis_label="Arrhythmia",
        premise_type="Diagnostic Criteria",
        diagnostic_path=("Cardiac", "Arrhythmia"),
        source_chunk_id="source-chunk:" + "2" * 20,
        knowledge_source_ids=("guideline:2",),
        source_origin="controlled_direct_graph",
    )
    retriever = IndependentFlatPremiseRetriever((other, gastritis))
    case = ClinicalCase(
        case_id="case-1",
        dataset="direct",
        task="diagnosis",
        sections={"history": "Epigastric pain with endoscopic inflammation."},
        gold_label="Gastritis",
    )

    bundle = retriever.retrieve(case, top_k=1)

    assert retriever.retriever_id == "controlled-premise-bm25-flat-v1"
    assert bundle.facts[0].source_chunk_id == SOURCE_ONE
    assert bundle.citation_allowlist == ()
