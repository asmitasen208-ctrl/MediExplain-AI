# Stage 6.3 educational intent dataset notes

## Purpose

This dataset is an initial, illustrative structure for a future five-category
text-classification demonstration. It does not train or evaluate a model and
is not ready for final ML evaluation.

## Provenance

The current examples are AI-generated synthetic educational sentences created
for this project. No external medical source, personal report or patient
information was used. This is not a declaration of professional authorship or
review.

The CSV contains the following metadata:

- `authorship_status`: currently `generated`.
- `review_status`: currently `unreviewed`.
- `review_date`: intentionally blank.
- `reviewer_role`: currently `none`.
- `source_note`: records that each sentence is a Stage 6.3 project-created
  illustration.

No source is claimed to have reviewed or endorsed any example.

## Category definitions

- `what_it_measures`: describes what a test or marker measures.
- `why_doctors_check_it`: explains why a clinician might obtain or review a test.
- `common_associations`: uses broad educational associations without patient
  diagnosis or treatment advice.
- `general_wellness`: discusses non-patient-specific wellness or preparation
  topics.
- `doctor_questions`: presents questions a person could ask a clinician.

The categories are semantically distinct in this initial set. Sentences remain
general and may require later human review for tone and factual precision.

## Grouping rules

Each `group_id` identifies one small semantic example family. Related
paraphrases or examples in the same family must retain the same `group_id` so
that future grouped splitting keeps them together.

The initial CSV uses test-specific family IDs across categories. A group may
therefore contain a measurement statement, a reason to check the test, an
association and a doctor question when all refer to the same test family:

- `haemoglobin_family` for haemoglobin examples.
- `glucose_family` for fasting glucose examples.
- `platelets_family` for platelet examples.
- `wbc_family` for white blood cell examples.
- `hba1c_family` for HbA1c examples.
- Other IDs describe general wellness or appointment topics.

Any future paraphrase must be assigned to an existing family only when it
conveys substantially the same idea. Genuinely different statements need a
different `group_id`.

## Safety and scope exclusions

The dataset excludes:

- Patient reports or personal medical data.
- Patient-specific diagnoses.
- Treatment recommendations.
- Medication names or doses.
- Clinical conclusions presented as individualized advice.
- Invented external citations.

## Review requirements

Before later model use:

- A qualified or designated human reviewer must independently review wording.
- Authors and reviewers must be identifiable, or dataset provenance may need
  to be kept at an explicitly fictional/demo level.
- The `review_status` and `review_date` fields must be updated only after actual
  review.
- Facts and category labels should be checked rather than inferred.
- Duplicate or near-duplicate rows should be removed.

No row in this initial dataset has received professional or human review.

## Expansion requirements

Future expansion should add genuinely distinct educational statements rather
than superficial word substitutions. It should be balanced where practical,
semantically varied and grouped before any future split.

Expansion must preserve every CSV column and supported label. It must not add
patient data or create an external source claim. Before model training, dataset
validation should verify balanced categories, unique text, valid groups and
compatibility with a reproducible stratified grouped split.
