# Project Plan

Use this file to align the group before implementation. Keep decisions concise
and cite sources as they are chosen.

## 1. Research Question

- Question:
- Why it goes beyond the obvious dataset task:
- Why it goes beyond merely comparing three algorithms:
- Planned insight or hypothesis:

## 2. Dataset(s)

- Dataset name:
- Public URL:
- Reference/citation:
- Dataset type: tabular / image / text / temporal / other
- Size check:
  - Instances/images:
  - Original features or image dimensions:
- Access notes:
- Local storage path: `data/raw/` (do not commit)

## 3. Feature Construction and Preprocessing

Describe the non-trivial transformations that will create meaningful inputs for
the algorithms.

- Method 1:
- Method 2:
- Method 3:
- Leakage risks and how they will be avoided:

## 4. Algorithms

Choose three distinct algorithms with different model class complexity.

| Complexity | Algorithm | Library or implementation plan | Key hyperparameters | Citation |
| --- | --- | --- | --- | --- |
| Simple | TBD | TBD | TBD | TBD |
| Medium | TBD | TBD | TBD | TBD |
| Complex | TBD | TBD | TBD | TBD |

Recent-paper algorithm requirement:

- Algorithm:
- First qualifying publication venue and year:
- Paper URL/DOI:
- Why it was not covered in class:

## 5. Outer Cross-Validation

- Technique: k-fold / repeated train-test / bootstrap / temporal split
- Repetitions/folds:
- Rationale:
- Implementation file:

## 6. Inner Nested Cross-Validation and Hyperparameter Tuning

- Technique:
- Repetitions/folds:
- Selection metric:

| Algorithm | Hyperparameter | Candidate values | Expected middle value |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |

## 7. Results To Report

At least three results are required, with error bars.

- Result 1:
- Result 2:
- Result 3:
- Error bar method:

## 8. Alternatives Considered

Record alternatives and why they were not selected. This will feed the critical
analysis section of the report.

| Alternative | Reason considered | Reason rejected |
| --- | --- | --- |
| TBD | TBD | TBD |
