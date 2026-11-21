# MediExplain AI

MediExplain AI is a Python 3.12 and Streamlit educational prototype for reviewing
blood laboratory reports and demonstrating a separate text-classification
machine-learning feature.

The application extracts report text, lets the user review extracted values,
and compares supplied report results with supplied laboratory reference ranges.
It also provides parameter-specific educational explanations in English and
Hinglish. The app is not a diagnostic or treatment system.

## Project objectives

- Provide a clear report-review workflow for pasted text, selectable PDFs,
  scanned PDFs and supported images.
- Preserve supplied report values until the user verifies and confirms them.
- Compare a result only with a reference range supplied in the same report
  when the units are compatible.
- Provide bounded, reviewed or fallback educational explanations without
  changing the interpretation classification.
- Demonstrate a separate TF-IDF and Logistic Regression text classifier for
  educational sentences.

## Technology stack

- Python 3.12
- Streamlit
- PyMuPDF for PDF text extraction and PDF page rendering
- Tesseract OCR through `pytesseract` for scanned PDF pages and images
- Pillow for image loading and conversion
- scikit-learn
- TF-IDF text vectorization
- Logistic Regression classification
- Joblib for saved-model loading

## Project structure

- `app.py` - Streamlit interface, report upload/extraction, manual review,
  confirmation snapshot, Phase 1 comparison, English/Hinglish explanations,
  and the ML Demonstration page.
- `train_ml_demo.py` - Explicit entry point for grouped training, inner model
  selection, outer held-out evaluation and model/artifact saving.
- `test_app_parser.py` - Unit and parser regression tests.
- `requirements.txt` - Python dependency list.
- `README.md` - Setup and project documentation.
- `modules/__init__.py` - Python package marker.
- `modules/blood_test_catalog.py` - Compatibility adapter for common test names,
  aliases, canonical names and panels.
- `modules/parameter_database.py` - Catalog loading, exact name resolution,
  unit compatibility checks and validated parameter lookup.
- `modules/report_interpreter.py` - Numeric/report range parsing and the
  deterministic Phase 1 classification engine. It does not modify confirmed
  report values.
- `modules/education_text_classifier.py` - Dataset validation, TF-IDF/Logistic
  Regression pipeline construction, input validation and model prediction helpers.
- `modules/interpretation_explainer.py` - English/Hinglish educational
  explanation selection and safe fallback handling.
- `data/blood_test_parameters.json` - Blood-test parameter catalog and
  laboratory/reference metadata.
- `data/parameter_education.json` - Reviewed or pending educational content,
  associations and general wellness text.
- `data/ml_education_intents.csv` - 25-row illustrative educational-text
  classification dataset.
- `data/ml_dataset_authoring_notes.md` - Dataset provenance, status,
  grouping rules and expansion requirements.
- `data/ml_model_evaluation.json` - Saved held-out and inner grouped evaluation
  metadata from the existing demonstration training run.
- `models/education_text_classifier.joblib` - Existing saved TF-IDF and
  Logistic Regression pipeline artifact.
- `assets/` - Reserved project asset directory.
- `.venv/` - Optional local virtual environment.

## Windows setup

Open PowerShell in the project directory and create a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks local activation scripts, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, without activation, commands can use the virtual-environment
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## Install dependencies

Install the dependencies from `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements include Streamlit, PyMuPDF, pytesseract, Pillow,
scikit-learn and Joblib.

## Optional Tesseract OCR setup

Pasted text and selectable-text PDFs continue to work without Tesseract.
Scanned PDF pages and uploaded images require the separate Tesseract
application.

1. Install Tesseract OCR for Windows.
2. Find the folder containing `tesseract.exe`.
3. Add that folder to the Windows `PATH` environment variable.
4. Restart PowerShell and restart the Streamlit process.
5. Verify the executable:

```powershell
tesseract --version
```

The application uses Tesseract through the Python `pytesseract` package. It
does not assume a fixed installation path.

## Launch the application

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The Streamlit app will provide the Dashboard, Upload Report, Analysis Results,
View All Results, ML Demonstration and other navigation destinations.

## Report workflow

1. Paste report text or upload a PDF, JPG, JPEG or PNG file.
2. Pasted text and selectable PDF text are extracted directly.
3. Scanned PDF pages and images use local Tesseract OCR.
4. Supported test names and values are converted into editable report rows.
5. The user may correct test names, results, units and reference ranges.
6. Missing or uncertain fields remain available for manual review.
7. The user explicitly confirms the verified row table.
8. The confirmed snapshot is used by the separate Phase 1 interpretation
   table.

The app does not automatically confirm extracted report information.

## Reference-range interpretation

For a confirmed row, Phase 1 can classify a numeric result as:

- Low
- High
- Within reference range
- Unable to classify

Comparison requires a matched parameter and a report-supplied range with
compatible units. Result and range units are not silently converted.

Unsupported names, units, missing fields or unsupported formats are marked for
manual review rather than guessed.

Parameter-specific explanations are shown in English or Hinglish. Educational
content cannot change the report interpretation classification or provide
diagnosis or treatment.

## ML Demonstration

The ML Demonstration is separate from report interpretation.

It uses:

- TF-IDF text vectorization
- Logistic Regression
- scikit-learn
- Joblib artifact loading

The five categories are:

- `what_it_measures`
- `why_doctors_check_it`
- `common_associations`
- `general_wellness`
- `doctor_questions`

The page loads `models/education_text_classifier.joblib` and accepts one
educational sentence entered in its own text area. The predicted category and
five model probability outputs are displayed only after the Predict button is
pressed. Probability values are model outputs, not calibrated confidence or
certainty.

The training dataset contains 25 generated, unreviewed educational examples.
It is an illustrative demo dataset, not a clinically reviewed or
representative real-world dataset.

The ML page does not access report text, uploaded files, OCR output, confirmed
report snapshots or report interpretation functions.

## Train or re-evaluate the ML demonstration

Training and evaluation are explicit operations in `train_ml_demo.py`.
Importing the script does not train or write model files.

Run it from the project directory only when an approved model update is
intended:

```powershell
.\.venv\Scripts\python.exe train_ml_demo.py
```

The script validates grouped outer and inner partitions, trains TF-IDF and
Logistic Regression pipelines within the development partition, performs one
outer held-out evaluation and writes:

- `models/education_text_classifier.joblib`
- `data/ml_model_evaluation.json`

The existing held-out evaluation is exploratory because the dataset is tiny
and all examples are generated and unreviewed.

## Tests

Run the discovered unittest suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

The current test file is `test_app_parser.py`. It contains report extraction
and unit-compatibility regression tests.

## Limitations and safety

- MediExplain AI is an educational prototype, not a diagnostic system.
- It does not replace a qualified clinician or laboratory professional.
- It does not diagnose conditions or recommend medicines, doses or treatment.
- Reference intervals require user-supplied laboratory ranges and compatible
  units.
- OCR can misread names, numbers, decimal points and units.
- Report values should always be manually verified before confirmation.
- The ML dataset contains 25 generated, unreviewed examples.
- No real-world or clinical accuracy is claimed for the text classifier.
- The held-out evaluation is exploratory and should not be presented as
  clinically validated.
- The vocabulary-coverage warning is advisory, not reliable out-of-scope
  detection.
- `Medicine Information` and `About Project` are currently placeholder
  navigation destinations.

## Prototype scope

The Blood Report and educational explanation workflows and the separate ML
Demonstration should remain isolated. The ML page must not change report
classification, OCR, confirmation snapshots or the existing parameter and
education databases.
