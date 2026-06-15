from entity_extraction.vocabularies import category_for


def main() -> None:
    checks = {
        "ICD10CM": "diagnosis",
        "RXNORM": "medication",
        "CPT": "therapy_or_procedure",
        "LNC": "lab_or_measurement",
        "MEDCIN": "therapy_or_procedure",
    }
    for source, expected in checks.items():
        actual = category_for(source)
        if actual != expected:
            raise AssertionError(f"{source}: expected {expected}, got {actual}")
    print("clinical vocabulary smoke test passed")


if __name__ == "__main__":
    main()
