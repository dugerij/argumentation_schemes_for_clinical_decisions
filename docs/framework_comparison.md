# Argumentation Framework Comparison

This repository currently implements an abstract argumentation baseline in [argumentation/aaf.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/argumentation/aaf.py:6) and a symbolic clinical argument generator in [argumentation/schemes.py](/Users/oluwatosinoso/Library/CloudStorage/OneDrive-hull.ac.uk/argumentation_schemes/argumentation/schemes.py:37).

## Current Baseline: Abstract Argumentation

Strengths:
- Simple and inspectable.
- Good first correctness baseline for attack/defense structure.
- Easy to test with grounded and preferred semantics.

Limitations:
- No graded confidence.
- No explicit patient-value tradeoffs.
- Competing arguments are either in or out.

Use it when:
- You need the first end-to-end working pipeline.
- You are still validating retrieval quality and attack construction.
- You want failure analysis before adding scoring complexity.

## Weighted Frameworks

What changes:
- Arguments or attacks get numerical strengths.
- Resolution can prefer stronger evidence or penalize weak support.

Best fit here:
- Retrieval already gives confidence-like signals such as repeated evidence support, source quality, or verifier acceptance rates.
- You want safer ranking among multiple acceptable treatments.

Expected implementation path:
1. Keep the current `Argument` structure.
2. Add explicit support/attack weights in metadata.
3. Introduce a weighted solver beside the current abstract solver rather than replacing it.

Main risk:
- Weight design can become arbitrary if the retrieval and verifier signals are not calibrated first.

## Value-Based Frameworks

What changes:
- Arguments are evaluated partly by the values they promote, such as safety, renal preservation, symptom relief, or guideline adherence.

Best fit here:
- The scenario explicitly includes a clinical goal and tradeoff.
- Different stakeholders or use cases need different value orderings.

Expected implementation path:
1. Normalize patient goals and safety priorities into a small value ontology.
2. Attach promoted/undermined values to arguments.
3. Add preference orderings over values at decision time.

Main risk:
- Requires a clear value model and more careful evaluation design than the current smoke-test setup.

## Recommendation

Recommended order for this repo:
1. Keep abstract argumentation as the correctness baseline.
2. Benchmark embedding models and retrieval quality first.
3. Add weighted argumentation once retrieval and verifier signals are stable.
4. Add value-based argumentation when patient-goal tradeoffs become a primary evaluation target.

In short:
- `abstract` is the right baseline now.
- `weighted` is the most practical next extension.
- `value-based` is likely the best long-term clinical decision layer, but only after the evidence and scoring signals are trustworthy.
