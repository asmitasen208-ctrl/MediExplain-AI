"""MediExplain AI: a lab report extraction prototype."""

from io import BytesIO
from pathlib import Path
import html
import re
import streamlit as st
import pymupdf
import joblib
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
import pytesseract
from modules.blood_test_catalog import LAB_TEST_NAME_PATTERN, TEST_DEFINITIONS
from modules.education_text_classifier import (
    SUPPORTED_CATEGORY_LABELS,
    get_pipeline_fit_warning,
    predict_education_category,
)
from modules.interpretation_explainer import build_educational_explanation
from modules.report_interpreter import interpret_report_rows as classify_report_rows


# Set the browser tab title and use a wide layout for the dashboard.
st.set_page_config(
    page_title="MediExplain AI",
    page_icon="🩺",
    layout="wide",
)

# Keep the prototype's visual styling in one place for easy beginner edits.
st.markdown(
    """
    <style>
    :root {
        color-scheme: light dark;
    }


    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: #f4f8fc;
        color: #263746;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #123b62;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #263746;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dce7f0;
        box-shadow: 3px 0 14px rgba(20, 54, 82, 0.04);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #263746;
    }

    .app-header {
        background: linear-gradient(115deg, #ffffff 35%, #eaf4fb 100%);
        border: 1px solid #dce8f1;
        border-radius: 18px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(20, 54, 82, 0.07);
    }

    .app-title {
        color: #123b62;
        font-size: 2rem;
        font-weight: 750;
        margin: 0;
    }

    .app-subtitle {
        color: #087487;
        font-size: 1.05rem;
        margin: 0.25rem 0 0;
    }

    .prototype-note {
        color: #465b6b;
        font-size: 0.82rem;
        margin-top: 0.7rem;
    }

    .section-card {
        background-color: #ffffff;
        border: 1px solid #dce8f1;
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        margin: 0.5rem 0 1rem;
        box-shadow: 0 6px 18px rgba(20, 54, 82, 0.06);
    }

    .section-card h3 {
        color: #123b62;
        margin-top: 0;
    }

    .success-message {
        background-color: #e8f5e9;
        border: 1px solid #81c784;
        border-radius: 10px;
        color: #1b5e20;
        margin: 0.5rem 0;
        padding: 0.75rem 1rem;
    }

    .success-message p {
        color: #1b5e20;
        margin: 0;
    }

    /* Keep all action buttons consistent and visibly interactive. */
    div.stButton > button {
        background-color: #1769aa;
        border: 1px solid #1769aa;
        border-radius: 10px;
        color: #ffffff;
        min-height: 2.65rem;
        font-weight: 600;
        box-shadow: 0 3px 9px rgba(23, 105, 170, 0.16);
    }

    div.stButton > button:hover {
        background-color: #12588f;
        border-color: #12588f;
        color: #ffffff;
    }

    div.stButton > button:focus-visible,
    [data-baseweb="tab"]:focus-visible,
    input:focus-visible,
    textarea:focus-visible {
        outline: 3px solid #80bce8;
        outline-offset: 2px;
    }

    /* File uploader dropzone uses a soft blue surface and readable text. */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #eef6fc;
        border: 1px dashed #7daed0;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(20, 54, 82, 0.04);
    }

    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] small {
        color: #263746;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background-color: #1769aa;
        border-color: #1769aa;
        border-radius: 8px;
        color: #ffffff;
    }

    /* Make tabs, field labels, and input text easy to distinguish. */
    button[data-baseweb="tab"] {
        color: #354b5d;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #123b62;
        border-bottom-color: #1769aa;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #263746;
        font-weight: 600;
    }

    .stTextInput input,
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #aebfce;
        border-radius: 9px;
        color: #263746;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #657887;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# The shared header stays visible whichever sidebar section is selected.
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">MediExplain AI</p>
        <p class="app-subtitle">Lab Report Simplifier &amp; Wellness Guide</p>
        <p class="prototype-note">
            Educational prototype only. This interface does not provide medical
            advice, diagnosis, or treatment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Simple sidebar navigation; each destination currently shows placeholder text.
navigation_items = [
    "Dashboard",
    "Upload Report",
    "Analysis Results",
    "View All Results",
    "Medicine Information",
    "ML Demonstration",
    "About Project",
]
selected_page = st.sidebar.radio("Navigation", navigation_items)


class ReportTextExtractionError(Exception):
    """Raised when report text cannot be safely extracted."""


class PDFTextExtractionError(ReportTextExtractionError):
    """Raised when a PDF cannot be safely converted to report text."""


class ImageTextExtractionError(ReportTextExtractionError):
    """Raised when an uploaded image cannot be converted to report text."""


class OCRUnavailableError(ReportTextExtractionError):
    """Raised when the separate Tesseract executable is not accessible."""


class OCRProcessingError(ReportTextExtractionError):
    """Raised when Tesseract cannot complete an OCR operation."""


_TESSERACT_UNAVAILABLE_MESSAGE = (
    "OCR is unavailable because Tesseract is not installed or accessible. "
    "Install the separate Tesseract OCR application, add the folder containing "
    "tesseract.exe to Windows PATH, then restart Streamlit. See README.md for setup."
)


def _has_meaningful_text(text: str) -> bool:
    """Ignore page markers, whitespace, and tiny fragments when deciding on OCR."""
    alphanumeric_count = len(re.findall(r"[A-Za-z0-9]", text))
    test_name_match = LAB_TEST_NAME_PATTERN.search(text)
    if test_name_match:
        remaining_text = (
            text[:test_name_match.start()] + text[test_name_match.end():]
        )
        if re.search(r"[0-9]", remaining_text):
            return True
    return (
        alphanumeric_count >= 8
        and len(re.findall(r"[A-Za-z]", text)) >= 3
    )


def _require_tesseract() -> None:
    """Check that the Tesseract executable can be launched without assuming a path."""
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as error:
        raise OCRUnavailableError(_TESSERACT_UNAVAILABLE_MESSAGE) from error
    except pytesseract.TesseractError as error:
        raise OCRProcessingError(
            "Tesseract is installed but could not be started. Check its installation "
            "and Windows PATH, then restart Streamlit."
        ) from error


def _ocr_image(image: Image.Image) -> str:
    """Return the best local Tesseract OCR candidate without correcting report values."""
    improved_text, _raw_text = _ocr_image_with_raw(image)
    return improved_text


def _ocr_image_raw(image: Image.Image) -> str:
    """Run unmodified Tesseract OCR once and retain its exact text output."""
    _require_tesseract()
    try:
        return pytesseract.image_to_string(
            image,
            config="--oem 3 --psm 6",
        )
    except pytesseract.TesseractNotFoundError as error:
        raise OCRUnavailableError(_TESSERACT_UNAVAILABLE_MESSAGE) from error
    except pytesseract.TesseractError as error:
        raise OCRProcessingError(
            "Tesseract could not read this page or image. Check the file and try again."
        ) from error
    except (OSError, RuntimeError) as error:
        raise OCRProcessingError(
            "An error occurred while running Tesseract OCR."
        ) from error


def _preprocess_ocr_image(image: Image.Image) -> Image.Image:
    """Upscale and enhance a report image for tabular-character recognition."""
    source = image.convert("RGB")
    scale = 2.0
    if max(source.size) < 2400:
        scaled = source.resize(
            (
                max(1, round(source.width * scale)),
                max(1, round(source.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        scaled = source

    grayscale = ImageOps.grayscale(scaled)
    grayscale = ImageOps.autocontrast(grayscale)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(1.35)
    return grayscale.filter(ImageFilter.SHARPEN)


def _ocr_candidate_score(text: str) -> int:
    """Rank OCR variants by recognized report structure, not by guessed values."""
    test_count = 0
    if "_FLEXIBLE_LAB_TEST_NAME_PATTERN" in globals():
        test_count = len(_FLEXIBLE_LAB_TEST_NAME_PATTERN.findall(text))

    numeric_count = len(re.findall(r"\d", text))
    bounded_range_count = len(
        re.findall(
            rf"{_RESULT_NUMBER_PATTERN}\s*(?:-|–|—|\bto\b)\s*"
            rf"{_RESULT_NUMBER_PATTERN}",
            text,
            re.IGNORECASE,
        )
    )
    score = (
        test_count * 100
        + bounded_range_count * 30
        + numeric_count * 2
    )
    if re.search(r"(?<![A-Za-z])/ul\b", text, re.IGNORECASE):
        score += 50
    if re.search(r"\bJul\b", text, re.IGNORECASE):
        score -= 20
    return score


def _ocr_image_with_raw(image: Image.Image) -> tuple[str, str]:
    """Return an improved OCR candidate and the exact original Tesseract text."""
    _require_tesseract()
    raw_text = _ocr_image_raw(image)
    candidates = [raw_text]

    preprocessed = _preprocess_ocr_image(image)
    try:
        processed_text = pytesseract.image_to_string(
            preprocessed,
            config="--oem 3 --psm 6",
        )
        candidates.append(processed_text)

        thresholded = preprocessed.point(
            lambda value: 0 if value < 180 else 255,
            "1",
        )
        threshold_text = pytesseract.image_to_string(
            thresholded,
            config="--oem 3 --psm 6",
        )
        candidates.append(threshold_text)
    except pytesseract.TesseractNotFoundError as error:
        raise OCRUnavailableError(_TESSERACT_UNAVAILABLE_MESSAGE) from error
    except pytesseract.TesseractError as error:
        raise OCRProcessingError(
            "Tesseract could not read this page or image. Check the file and try again."
        ) from error
    except (OSError, RuntimeError) as error:
        raise OCRProcessingError(
            "An error occurred while running Tesseract OCR."
        ) from error

    scored_candidates = [
        (_ocr_candidate_score(candidate), index, candidate)
        for index, candidate in enumerate(candidates)
    ]
    _score, _index, improved_text = max(
        scored_candidates,
        key=lambda item: (item[0], -item[1]),
    )
    return improved_text, raw_text


def _ocr_pdf_page(page: pymupdf.Page) -> str:
    """Render one PDF page at a readable resolution and OCR it locally."""
    pixmap = None
    image = None
    try:
        pixmap = page.get_pixmap(
            dpi=225,
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        return _ocr_image(image)
    except (OSError, ValueError, RuntimeError) as error:
        raise PDFTextExtractionError(
            "A scanned PDF page could not be rendered for OCR."
        ) from error
    finally:
        if image is not None:
            image.close()
        if pixmap is not None:
            del pixmap


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text and OCR only pages without meaningful text."""
    if not pdf_bytes:
        raise PDFTextExtractionError("The selected PDF is empty.")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (
        pymupdf.EmptyFileError,
        pymupdf.FileDataError,
        ValueError,
        RuntimeError,
    ) as error:
        raise PDFTextExtractionError(
            "The selected file is not a readable PDF or is corrupted."
        ) from error

    try:
        if document.needs_pass:
            raise PDFTextExtractionError(
                "This PDF is password-protected. Remove its password and try again."
            )
        if document.page_count == 0:
            raise PDFTextExtractionError("The selected PDF contains no pages.")

        page_sections = []
        has_meaningful_text = False
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if not _has_meaningful_text(page_text):
                page_text = _ocr_pdf_page(page).strip()
                if not _has_meaningful_text(page_text):
                    raise PDFTextExtractionError(
                        f"OCR did not recognize usable text on PDF page "
                        f"{page_number}; existing report rows were not changed."
                    )
            if _has_meaningful_text(page_text):
                has_meaningful_text = True
            page_sections.append(f"--- Page {page_number} ---\n{page_text}")

        if not has_meaningful_text:
            raise PDFTextExtractionError(
                "No readable text was found in the PDF. OCR did not recognize "
                "usable text; verify the scan quality and try again."
            )
        return "\n\n".join(page_sections)
    except (
        pymupdf.EmptyFileError,
        pymupdf.FileDataError,
        ValueError,
        RuntimeError,
    ) as error:
        raise PDFTextExtractionError(
            "Text could not be safely extracted from this PDF."
        ) from error
    finally:
        document.close()


def extract_image_text(image_bytes: bytes) -> str:
    """Open a supported image and return its unmodified local OCR output."""
    if not image_bytes:
        raise ImageTextExtractionError("The selected image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            if source_image.format not in {"JPEG", "PNG"}:
                raise ImageTextExtractionError(
                    "The selected file is not a supported JPEG or PNG image."
                )
            source_image.load()
            image = source_image.convert("RGB")
            try:
                recognized_text, raw_ocr_text = _ocr_image_with_raw(image)
                st.session_state["raw_ocr_text_debug"] = raw_ocr_text
            finally:
                image.close()
    except Image.DecompressionBombError as error:
        raise ImageTextExtractionError(
            "The selected image is too large to process safely."
        ) from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise ImageTextExtractionError(
            "The selected image is empty, corrupted, or unsupported."
        ) from error

    if not _has_meaningful_text(recognized_text):
        raise ImageTextExtractionError(
            "OCR did not recognize usable text in the image. Try a clearer scan "
            "or paste the report text."
        )
    return recognized_text


_FLEXIBLE_LAB_TEST_NAME_PATTERN = re.compile(
    LAB_TEST_NAME_PATTERN.pattern.replace(r"\s+", r"[\W_]*"),
    LAB_TEST_NAME_PATTERN.flags,
)
_RESULT_NUMBER_PATTERN = (
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?"
    r"|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
_RESULT_LABEL_PATTERN = re.compile(
    r"^[ \t]*[:=]?[ \t]*(?:Observed[ \t]+Value|Result[ \t]+Value|Result|Value)"
    r"\b[ \t]*[:=]?[ \t]*(?P<value>[^\r\n]*)",
    re.IGNORECASE | re.MULTILINE,
)
_UNIT_LABEL_PATTERN = re.compile(
    r"^[ \t]*[:=]?[ \t]*Unit\b[ \t]*[:=]?[ \t]*(?P<value>[^\r\n]*)",
    re.IGNORECASE | re.MULTILINE,
)
_REFERENCE_RANGE_LABEL_PATTERN = re.compile(
    r"(?<!\w)(?:Reference[ \t]*Range|Normal[ \t]*Range|Ref\.?[ \t]*Range)"
    r"\b[ \t]*[:=]?[ \t]*(?P<value>[^\r\n]*)",
    re.IGNORECASE,
)
_KNOWN_LAB_UNIT_KEYS = {
    re.sub(r"\s+", "", unit).casefold()
    for definition in TEST_DEFINITIONS
    for unit in definition.units
}

_REPORT_WARNING_BACKGROUND = "#FFF4CC"
_REPORT_WARNING_TEXT = "#333333"
_REPORT_WARNING_BORDER = "#E0B84C"


def _normalize_ocr_per_ul_unit(test_name: str, unit: str) -> str:
    """Normalize only obvious WBC or Platelet OCR forms of /uL."""
    normalized_test_name = re.sub(r"[^a-z0-9]+", "", test_name.casefold())
    if normalized_test_name not in {"wbc", "platelets", "platelet"}:
        return unit
    if not isinstance(unit, str):
        return unit

    compact_unit = re.sub(r"\s+", "", unit).casefold()
    compact_unit = compact_unit.replace("µ", "u").replace("μ", "u")
    if compact_unit in {"jul", "/ul", "j/ul"}:
        return "/uL"
    return unit


def _render_report_warning(message: str) -> None:
    """Render the two report-review warning strips with explicit readable colors."""
    escaped_message = html.escape(str(message))
    escaped_message = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        escaped_message,
    )
    escaped_message = escaped_message.replace("\n", "<br>")
    st.markdown(
        f"""
        <div role="alert" style="
            background-color: {_REPORT_WARNING_BACKGROUND};
            color: {_REPORT_WARNING_TEXT};
            border: 1px solid {_REPORT_WARNING_BORDER};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
        ">
            <span style="color: {_REPORT_WARNING_TEXT};" aria-hidden="true">⚠</span>
            <span style="color: {_REPORT_WARNING_TEXT};">{escaped_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _is_plausible_lab_unit(unit: str) -> bool:
    """Accept catalog units and compact unit-like expressions, not prose."""
    normalized_unit = re.sub(r"\s+", "", unit).casefold()
    return normalized_unit in _KNOWN_LAB_UNIT_KEYS or bool(
        re.fullmatch(
            r"(?:/[a-z0-9µμ().]+|"
            r"[a-z0-9µμ%]+(?:[/^*×][a-z0-9µμ().]+)+)",
            normalized_unit,
        )
    )


def _extract_result_and_unit(
    test_section: str,
) -> tuple[str, str, bool, int | None]:
    """Extract a numeric result and its adjacent unit without converting either."""
    lines = test_section.splitlines()
    result_label_match = _RESULT_LABEL_PATTERN.search(test_section)
    if result_label_match:
        result_text = result_label_match.group("value").strip()
        result_line_number = test_section[:result_label_match.start()].count("\n")
        if not result_text:
            for line_number in range(result_line_number + 1, len(lines)):
                next_line = lines[line_number].strip()
                if not next_line:
                    continue
                if (
                    _RESULT_LABEL_PATTERN.match(next_line)
                    or _UNIT_LABEL_PATTERN.match(next_line)
                    or _REFERENCE_RANGE_LABEL_PATTERN.search(next_line)
                ):
                    break
                result_text = next_line
                break
        parsed_result = _parse_result_line(result_text)
        if parsed_result is None:
            range_label_match = _REFERENCE_RANGE_LABEL_PATTERN.search(result_text)
            if range_label_match:
                result_text = result_text[:range_label_match.start()].strip()
            if result_text:
                return result_text, "Unavailable", True, result_line_number
            return "Unavailable", "Unavailable", True, result_line_number
        result, unit = parsed_result
        return result, unit, True, result_line_number

    pending_result_marker = False
    for line_number, line in enumerate(lines):
        candidate = line.strip()
        if not candidate or candidate in {":", "=", "-"}:
            if candidate in {":", "="}:
                pending_result_marker = True
            continue
        if (
            _UNIT_LABEL_PATTERN.match(candidate)
            or _REFERENCE_RANGE_LABEL_PATTERN.search(candidate)
        ):
            return "Unavailable", "Unavailable", False, None
        parsed_result = _parse_result_line(candidate)
        if parsed_result is not None:
            result, unit = parsed_result
            return result, unit, False, line_number
        inline_value_match = re.match(
            r"^[ \t]*[:=][ \t]*(?P<value>[A-Za-z][A-Za-z0-9+<>._-]*)[ \t]*$",
            candidate,
        )
        if inline_value_match:
            return inline_value_match.group("value"), "Unavailable", False, line_number
        if pending_result_marker and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9+<>._-]*", candidate
        ):
            return candidate, "Unavailable", False, line_number
        return "Unavailable", "Unavailable", False, None

    return "Unavailable", "Unavailable", False, None


def _parse_result_line(result_text: str) -> tuple[str, str] | None:
    """Parse a result token and retain only a plausible adjacent unit."""
    range_label_match = _REFERENCE_RANGE_LABEL_PATTERN.search(result_text)
    if range_label_match:
        result_text = result_text[:range_label_match.start()]
    inline_range_match = re.search(
        rf"(?<![0-9])(?P<range>"
        rf"{_RESULT_NUMBER_PATTERN}\s*(?:-|–|—|\bto\b)\s*"
        rf"{_RESULT_NUMBER_PATTERN}(?:\s*(?:[A-Za-zµμ/%][A-Za-z0-9µμ/%^*×./_-]*))?"
        rf"|(?:below|under|less\s+than|above|over|greater\s+than)\s*"
        rf"{_RESULT_NUMBER_PATTERN}(?:\s*(?:[A-Za-zµμ/%][A-Za-z0-9µμ/%^*×./_-]*))?)",
        result_text,
        re.IGNORECASE,
    )
    if inline_range_match:
        result_text = (
            result_text[:inline_range_match.start()]
            + " "
            + result_text[inline_range_match.end():]
        )
    result_match = re.match(
        rf"^[ \t]*[:=]?[ \t]*(?P<result>{_RESULT_NUMBER_PATTERN})(?P<tail>.*)$",
        result_text.strip(),
    )
    if not result_match:
        return None

    unit_text = result_match.group("tail").strip()
    inline_unit_match = re.match(
        r"^Unit\b[ \t]*[:=]?[ \t]*(.*)$",
        unit_text,
        re.IGNORECASE,
    )
    if inline_unit_match:
        unit_text = inline_unit_match.group(1).strip()
    if not unit_text:
        return result_match.group("result"), "Unavailable"
    if _is_plausible_lab_unit(unit_text):
        return result_match.group("result"), unit_text
    if re.fullmatch(
        r"[A-Za-zµμ%]+(?:[/^*×][A-Za-z0-9µμ().]+)*",
        unit_text,
    ):
        return result_match.group("result"), unit_text
    return result_match.group("result"), "Unavailable"


def _extract_labeled_value(
    test_section: str,
    label_pattern: re.Pattern[str],
) -> tuple[str, bool]:
    """Read a labeled value from its line or the following non-label line."""
    label_match = label_pattern.search(test_section)
    if not label_match:
        return "Unavailable", False

    value = label_match.group("value").strip()
    if value:
        return value, True

    for line in test_section[label_match.end():].splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if (
            _RESULT_LABEL_PATTERN.match(candidate)
            or _UNIT_LABEL_PATTERN.match(candidate)
            or _REFERENCE_RANGE_LABEL_PATTERN.search(candidate)
        ):
            break
        return candidate, True
    return "Unavailable", True


def _extract_reference_range_from_row(test_section: str) -> str:
    """Find a range only inside the current test row, never in another row."""
    if re.search(
        r"(?im)^[ \t]*(?:REFERENCE[ \t]+RANGE|REFERENCE[ \t]*RANGE)"
        r"[ \t]*:?[ \t]*$",
        test_section,
    ):
        return "Unavailable"
    labeled_range, has_label = _extract_labeled_value(
        test_section,
        _REFERENCE_RANGE_LABEL_PATTERN,
    )
    if has_label:
        return labeled_range

    threshold_match = re.search(
        rf"(?<!\w)(?P<range>"
        rf"(?:below|under|less\s+than)\s*{_RESULT_NUMBER_PATTERN}"
        rf"(?:\s*(?:[A-Za-zµμ/%][A-Za-z0-9µμ/%^*×./_-]*))?)",
        test_section,
        re.IGNORECASE,
    )
    if threshold_match:
        return re.sub(r"\s+", " ", threshold_match.group("range")).strip()

    bounded_match = re.search(
        rf"(?<![0-9])(?P<low>{_RESULT_NUMBER_PATTERN})"
        rf"\s*(?:-|–|—|\bto\b)\s*"
        rf"(?P<high>{_RESULT_NUMBER_PATTERN})"
        rf"(?:\s*(?P<unit>[A-Za-zµμ/%][A-Za-z0-9µμ/%^*×./_-]*))?",
        test_section,
        re.IGNORECASE,
    )
    if bounded_match:
        low = bounded_match.group("low")
        high = bounded_match.group("high")
        unit = bounded_match.group("unit")
        if unit:
            return f"{low} - {high} {unit}"
        return f"{low} - {high}"

    return "Unavailable"


_REFERENCE_RANGE_SECTION_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:REFERENCE[ \t]+RANGE|REFERENCE[ \t]*RANGE)"
    r"[ \t]*:?[ \t]*$",
)


def _is_valid_reference_range_entry(entry: str) -> bool:
    """Accept only bounded or one-sided range entries, not arbitrary report lines."""
    number = _RESULT_NUMBER_PATTERN
    bounded_match = re.fullmatch(
        rf"[ \t]*(?P<low>{number})[ \t]*(?:-|–|—|to)[ \t]*"
        rf"(?P<high>{number})"
        rf"[ \t]*(?P<unit>[A-Za-zµμ%]+(?:[/^*×][A-Za-z0-9µμ().]+)*)?[ \t]*",
        entry,
        re.IGNORECASE,
    )
    if bounded_match:
        return True

    threshold_match = re.fullmatch(
        rf"[ \t]*(?:below|under|less[ \t]+than|above|over|"
        rf"greater[ \t]+than|<=|>=|<|>)[ \t]*"
        rf"{number}"
        rf"[ \t]*(?:[A-Za-zµμ%]+(?:[/^*×][A-Za-z0-9µμ().]+)*)?[ \t]*",
        entry,
        re.IGNORECASE,
    )
    return threshold_match is not None


def _extract_positional_reference_ranges(
    report_text: str,
) -> tuple[list[str] | None, bool]:
    """Extract a clearly labelled range section only as an all-or-nothing mapping."""
    section_match = _REFERENCE_RANGE_SECTION_PATTERN.search(report_text)
    if section_match is None:
        return None, False

    lines = report_text[section_match.end():].splitlines()
    entries: list[str] = []
    ambiguous = False

    for line in lines:
        candidate = line.strip()
        if not candidate:
            continue
        if re.fullmatch(r"-{2,}|={2,}|_{2,}", candidate):
            continue
        if re.search(r":", candidate):
            break
        if re.fullmatch(
            r"[A-Z][A-Z0-9 \t/&()._-]*",
            candidate,
            re.IGNORECASE,
        ) and not _is_valid_reference_range_entry(candidate):
            break
        if not _is_valid_reference_range_entry(candidate):
            ambiguous = True
            continue
        entries.append(re.sub(r"[ \t]+", " ", candidate).strip())

    if ambiguous:
        return None, True
    if not entries:
        return None, True
    return entries, True


def extract_lab_information(report_text: str) -> list[dict[str, str]]:
    """Extract supported report rows while leaving uncertain fields unavailable."""
    test_matches = list(_FLEXIBLE_LAB_TEST_NAME_PATTERN.finditer(report_text))
    extracted_rows = []

    for index, match in enumerate(test_matches):
        section_end = (
            test_matches[index + 1].start()
            if index + 1 < len(test_matches)
            else len(report_text)
        )
        test_section = report_text[match.end():section_end]
        result, inferred_unit, has_result_label, result_line_number = (
            _extract_result_and_unit(test_section)
        )
        unit, has_unit_label = _extract_labeled_value(
            test_section,
            _UNIT_LABEL_PATTERN,
        )
        if unit == "Unavailable":
            unit = inferred_unit
        reference_range = _extract_reference_range_from_row(test_section)
        has_reference_label = reference_range != "Unavailable"

        has_row_marker = bool(
            re.match(r"^[ \t]*[:=][ \t]*(?:\r?\n|$)", test_section)
        )
        if not (
            result != "Unavailable"
            or has_result_label
            or has_unit_label
            or has_reference_label
            or has_row_marker
        ):
            continue

        if unit == "Unavailable" and result_line_number is not None:
            lines = test_section.splitlines()
            for line in lines[result_line_number + 1:]:
                candidate = line.strip()
                if not candidate:
                    continue
                if (
                    _RESULT_LABEL_PATTERN.match(candidate)
                    or _UNIT_LABEL_PATTERN.match(candidate)
                    or _REFERENCE_RANGE_LABEL_PATTERN.search(candidate)
                ):
                    break
                if _is_plausible_lab_unit(candidate):
                    unit = candidate
                break

        unit = _normalize_ocr_per_ul_unit(match.group("name").strip(), unit)

        extracted_rows.append(
            {
                "Test name": match.group("name").strip(),
                "Result value": result,
                "Unit": unit,
                "Reference range": reference_range,
            }
        )

    positional_ranges, has_positional_section = (
        _extract_positional_reference_ranges(report_text)
    )
    if (
        has_positional_section
        and positional_ranges is not None
        and len(extracted_rows) == len(positional_ranges)
    ):
        for row, positional_range in zip(extracted_rows, positional_ranges):
            if row["Reference range"] == "Unavailable":
                row["Reference range"] = positional_range

    return extracted_rows


def _is_missing_report_value(value: object) -> bool:
    """Treat empty values and the parser's missing-value marker as absent."""
    return value is None or str(value).strip().casefold() in {"", "unavailable"}


def _validate_report_rows(
    rows: list[dict[str, str]],
) -> list[tuple[int, list[str]]]:
    """Return row numbers and review messages without changing report values."""
    numeric_result_pattern = (
        r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?"
        r"|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    )
    normalized_names = []
    name_counts: dict[str, int] = {}

    for row in rows:
        name = row.get("Test name")
        if _is_missing_report_value(name):
            normalized_names.append(None)
            continue
        normalized_name = re.sub(r"\s+", " ", str(name)).strip().casefold()
        normalized_names.append(normalized_name)
        name_counts[normalized_name] = name_counts.get(normalized_name, 0) + 1

    issues_by_row = []
    for row_number, row in enumerate(rows, start=1):
        issues = []
        if _is_missing_report_value(row.get("Test name")):
            issues.append("Test name is missing.")
        if _is_missing_report_value(row.get("Result value")):
            issues.append("Result value is missing.")
        elif not re.fullmatch(
            numeric_result_pattern, str(row.get("Result value")).strip()
        ):
            issues.append("Result value is not a recognizable number.")
        if _is_missing_report_value(row.get("Unit")):
            issues.append("Unit is missing.")
        elif not _is_plausible_lab_unit(str(row.get("Unit")).strip()):
            suspicious_unit = str(row.get("Unit")).strip()
            issues.append(
                f"Unit '{suspicious_unit}' is suspicious or "
                "unrecognized; verify manually."
            )
        if _is_missing_report_value(row.get("Reference range")):
            issues.append("Reference range is missing.")

        normalized_name = normalized_names[row_number - 1]
        if normalized_name and name_counts[normalized_name] > 1:
            issues.append("Test name is duplicated in another row.")
        if issues:
            issues_by_row.append((row_number, issues))

    return issues_by_row


def build_interpretation_note(
    source_row: dict[str, str],
    interpretation_row: dict[str, str],
) -> str:
    """Make each comparison note specific to its supplied values and range."""
    def field(name: str) -> str:
        value = source_row.get(name)
        return "Unavailable" if value is None or not str(value).strip() else str(value)

    raw_name = field("Test name")
    result_value = field("Result value")
    result_unit = field("Unit")
    reference_range = field("Reference range")
    normalized_name = (
        interpretation_row.get("Normalized test name")
        or raw_name
    )
    classification = str(
        interpretation_row.get("Classification") or "Unable to classify"
    )
    original_note = str(
        interpretation_row.get("Notes")
        or "Interpretation details are unavailable."
    )
    note_key = original_note.casefold()

    if classification == "Unable to classify":
        return original_note

    comparison = (
        f"{normalized_name}: {result_value} {result_unit} is "
        f"{classification.casefold()} against the lab range "
        f"{reference_range}."
    )
    if "unit is omitted" in note_key:
        return (
            f"{comparison} The range has no unit, so the result unit is "
            "used for this row without conversion; verify the report's convention."
        )
    return comparison


def interpret_confirmed_report_rows(
    confirmed_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Interpret confirmed values and add notes that cite their exact inputs."""
    interpretation_rows = classify_report_rows(confirmed_rows)
    for source_row, interpretation_row in zip(
        confirmed_rows, interpretation_rows
    ):
        interpretation_row["Notes"] = build_interpretation_note(
            source_row,
            interpretation_row,
        )
    return interpretation_rows


def render_interpretation_table():
    """Display comparisons using only the confirmed report snapshot."""
    phase1_confirmed_rows = st.session_state.get("phase1_confirmed_report_rows")
    if phase1_confirmed_rows is None:
        st.info("Confirm report information to view the reference-range comparison.")
        return
    if (
        st.session_state.get("phase1_confirmed_report_version")
        != st.session_state.get("report_extraction_version")
    ):
        st.info(
            "The current report has not been confirmed yet. The previous "
            "confirmed report is retained but is not used for this comparison."
        )
        return
    if not phase1_confirmed_rows:
        st.info("The confirmed report contains no test rows to compare.")
        return

    st.caption(
        "Phase 1 compares results only with reference ranges supplied in this "
        "report. It does not use universal cutoffs or provide diagnoses or "
        "treatment recommendations."
    )
    interpretation_rows = interpret_confirmed_report_rows(phase1_confirmed_rows)

    selected_language = st.selectbox(
        "Explanation language",
        ("English", "Hinglish"),
        key="phase1_explanation_language",
    )
    educational_explanations = [
        build_educational_explanation(
            interpretation_row,
            language=selected_language,
        )
        for interpretation_row in interpretation_rows
    ]
    educational_explanations_available = sum(
        1
        for interpretation_row, explanation in zip(
            interpretation_rows,
            educational_explanations,
        )
        if (
            isinstance(interpretation_row.get("Parameter ID"), str)
            and bool(interpretation_row["Parameter ID"].strip())
            and interpretation_row.get("Resolution status") == "matched"
            and bool(explanation.get("parameter_name"))
            and explanation.get("content_review_status")
            not in {
                "unavailable",
                "invalid_content",
                "missing",
                "not_loaded",
            }
        )
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("Total tests", len(interpretation_rows))
    summary_columns[1].metric(
        "Within reference range",
        sum(
            row.get("Classification") == "Within reference range"
            for row in interpretation_rows
        ),
    )
    summary_columns[2].metric(
        "Unable to classify",
        sum(
            row.get("Classification") == "Unable to classify"
            for row in interpretation_rows
        ),
    )
    summary_columns[3].metric(
        "Educational explanations available",
        educational_explanations_available,
    )

    st.dataframe(
        interpretation_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Parameter ID": None,
            "Resolution status": None,
        },
    )

    for explanation in educational_explanations:
        content_review_status = explanation.get("content_review_status")
        fallback_message = explanation.get("fallback_message")

        if content_review_status == "not_loaded":
            with st.container(border=True):
                st.markdown(
                    "**Parameter-specific education is unavailable**"
                )
                st.markdown(
                    "It cannot be shown because the test identity or unit was "
                    "not fully resolved. The exact interpreter reason is in "
                    "the Notes column above."
                )
            continue

        if content_review_status in {
            "unavailable",
            "invalid_content",
            "missing",
        }:
            with st.container(border=True):
                st.markdown("**Educational information is unavailable**")
                st.markdown(
                    fallback_message
                    or "Educational information is unavailable for this result."
                )
            continue

        parameter_name = explanation.get("parameter_name")
        if not parameter_name:
            with st.container(border=True):
                st.markdown("**Educational information is unavailable**")
                st.markdown(
                    "Parameter-specific education is not available for this result."
                )
            continue

        with st.container(border=True):
            st.subheader(parameter_name)

            content_notice = explanation.get("content_notice")
            if content_notice:
                st.markdown("**Content review notice**")
                st.markdown(content_notice)

            if explanation.get("what_it_measures"):
                st.markdown("**What it measures**")
                st.write(explanation["what_it_measures"])

            if explanation.get("why_doctors_check_it"):
                st.markdown("**Why doctors check it**")
                st.write(explanation["why_doctors_check_it"])

            relevant_associations = explanation.get("relevant_associations")
            if relevant_associations:
                st.markdown("**Relevant associations**")
                for association in relevant_associations:
                    st.write(f"- {association}")

            if explanation.get("general_wellness"):
                st.markdown("**General wellness information**")
                st.write(explanation["general_wellness"])

            doctor_questions = explanation.get("doctor_questions")
            if doctor_questions:
                st.markdown("**Questions to ask your doctor**")
                for question in doctor_questions:
                    st.write(f"- {question}")

            sources = explanation.get("sources")
            if sources:
                st.markdown("**Sources**")
                for source in sources:
                    title = source.get("title")
                    url = source.get("url")
                    if title:
                        st.markdown(f"**{title}**")
                    if url:
                        st.markdown(f"[{url}]({url})")


def confirm_report_information(verified_report_rows: list[dict[str, str]]) -> None:
    """Save an independent snapshot of the current verified editor rows."""
    confirmed_snapshot = [dict(row) for row in verified_report_rows]
    st.session_state["edited_extracted_report_rows"] = [
        dict(row) for row in confirmed_snapshot
    ]
    st.session_state["phase1_confirmed_report_rows"] = confirmed_snapshot
    st.session_state["phase1_confirmed_report_version"] = st.session_state[
        "report_extraction_version"
    ]


def render_report_workspace():
    """Show the upload, pasted-text, and manual-entry layout."""
    st.markdown(
        """
        <div class="section-card">
            <h3>Report upload area</h3>
            <p>
                Add a report or paste its text. Files are processed only when
                you click Extract Report Information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a lab report",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Supported formats: PDF, JPG, JPEG, and PNG.",
    )
    is_pdf_upload = (
        uploaded_file is not None
        and uploaded_file.name.casefold().endswith(".pdf")
    )
    is_image_upload = (
        uploaded_file is not None
        and uploaded_file.name.casefold().endswith((".jpg", ".jpeg", ".png"))
    )
    if uploaded_file is not None:
        st.caption(
            "Selected file: "
            f"{uploaded_file.name} ({uploaded_file.size:,} bytes)."
        )
        if is_pdf_upload:
            st.caption(
                "Selectable PDF text is extracted directly; scanned pages use "
                "local Tesseract OCR."
            )
        elif is_image_upload:
            st.markdown(
                "**Image text is read locally with Tesseract OCR.**"
            )
        _render_report_warning(
            "If OCR is used, it can misread characters, decimal points, units, "
            "and reference ranges. Review and correct every extracted row before "
            "confirming."
        )

    # Initialize the widget state once so text remains editable across reruns.
    if "report_text" not in st.session_state:
        st.session_state["report_text"] = ""
    if st.button("Load Sample Report"):
        st.session_state["report_text"] = (
            "Patient ID: DEMO001\n"
            "Report Type: Blood Test\n"
            "Sample Type: Blood\n"
            "\n"
            "Haemoglobin: 14.2 g/dL\n"
            "Reference Range: 13.0 - 17.0\n"
            "\n"
            "WBC: 7200 /uL\n"
            "Reference Range: 4000 - 11000\n"
            "\n"
            "Platelets: 250000 /uL\n"
            "Reference Range: 150000 - 450000\n"
            "\n"
            "RBC: 4.8 million/uL\n"
            "Reference Range: 4.5 - 5.9\n"
            "\n"
            "Blood Glucose (Fasting): 145 mg/dL\n"
            "Reference Range: 70 - 99\n"
            "\n"
            "HbA1c: 6.8 %\n"
            "Reference Range: Below 5.7"
        )

    text_tab, manual_tab = st.tabs(["Paste report text", "Manual entry"])

    with text_tab:
        st.text_area(
            "Paste report text",
            key="report_text",
            height=170,
            placeholder="Paste report text here...",
        )

    with manual_tab:
        st.caption("Fields are provided for layout testing; no interpretation is performed.")
        first_row = st.columns(2)
        first_row[0].text_input("Test name", placeholder="For example, test name")
        first_row[1].text_input("Result", placeholder="Enter a result")
        second_row = st.columns(2)
        second_row[0].text_input("Unit", placeholder="Enter a unit")
        second_row[1].text_input("Reference range", placeholder="Enter a range")

    if "report_extraction_version" not in st.session_state:
        st.session_state["report_extraction_version"] = 0

    if st.button("Extract Report Information", type="primary", use_container_width=True):
        extracted_rows = None
        if uploaded_file is None:
            extracted_rows = extract_lab_information(
                st.session_state.get("report_text", "")
            )
        elif is_pdf_upload:
            try:
                report_text = extract_pdf_text(uploaded_file.getvalue())
            except ReportTextExtractionError as error:
                st.error(str(error))
            else:
                pdf_rows = extract_lab_information(report_text)
                if pdf_rows:
                    extracted_rows = pdf_rows
                else:
                    st.error(
                        "No supported blood tests were found in the PDF text. "
                        "Existing report rows were not changed."
                    )
        elif is_image_upload:
            try:
                report_text = extract_image_text(uploaded_file.getvalue())
            except ReportTextExtractionError as error:
                st.error(str(error))
            else:
                with st.expander(
                    "Raw OCR Text (Debug) - temporary debugging information",
                    expanded=False,
                ):
                    st.code(
                        st.session_state.get(
                            "raw_ocr_text_debug",
                            report_text,
                        ),
                        language=None,
                    )
                image_rows = extract_lab_information(report_text)
                if image_rows:
                    extracted_rows = image_rows
                else:
                    st.error(
                        "No supported blood tests were found in the OCR text. "
                        "Existing report rows were not changed."
                    )
        elif not is_image_upload:
            st.error("This file type is not supported for text extraction.")

        # The button runs once per click; upload selection and other reruns do not reprocess it.
        if extracted_rows is not None:
            st.session_state["original_extracted_report_rows"] = [
                dict(row) for row in extracted_rows
            ]
            st.session_state["report_extraction_version"] += 1
            st.session_state["edited_extracted_report_rows"] = [
                dict(row) for row in extracted_rows
            ]

    if "original_extracted_report_rows" in st.session_state:
        extracted_rows = st.session_state["original_extracted_report_rows"]
        if not extracted_rows:
            st.error(
                "No supported lab tests were found. Add report text with one "
                "of the supported test names, then try again."
            )
        else:
            st.subheader("Review extracted information")
            st.caption(
                "Verify or correct every field, then confirm the report. Only "
                "confirmed rows are used in the separate range comparison."
            )
            # A new extraction gets a fresh editor key; edits persist under it
            # across reruns without writing to the editor's session-state key.
            editor_key = (
                "report_rows_editor_phase1_"
                f"{st.session_state['report_extraction_version']}"
            )
            edited_report_rows = st.data_editor(
                st.session_state["edited_extracted_report_rows"],
                key=editor_key,
                hide_index=True,
                use_container_width=True,
                num_rows="dynamic",
            )
            verified_report_rows = [dict(row) for row in edited_report_rows]
            st.session_state["edited_extracted_report_rows"] = [
                dict(row) for row in verified_report_rows
            ]

            validation_issues = _validate_report_rows(verified_report_rows)
            if validation_issues:
                warning_lines = [
                    f"- Row {row_number}: {' '.join(issues)}"
                    for row_number, issues in validation_issues
                ]
                _render_report_warning(
                    f"**{len(validation_issues)} row(s) need review.**\n\n"
                    + "\n".join(warning_lines)
                )
            else:
                st.success(
                    "All rows pass the basic completeness and numeric checks."
                )

            if st.button("Confirm Report Information", type="primary"):
                confirm_report_information(verified_report_rows)

            confirmed_rows = st.session_state.get("phase1_confirmed_report_rows")
            if confirmed_rows is not None:
                is_current_confirmation = (
                    st.session_state.get("phase1_confirmed_report_version")
                    == st.session_state["report_extraction_version"]
                )
                if is_current_confirmation and confirmed_rows == verified_report_rows:
                    st.markdown(
                        """
                        <div class="success-message" role="status">
                            <p>Report information confirmed and saved.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(
                        "There are unconfirmed changes. Confirm the current "
                        "table to save the updated report information."
                    )

    if "phase1_confirmed_report_rows" in st.session_state:
        st.subheader("Phase 1: Reference-range comparison")
        render_interpretation_table()


class MLArtifactError(Exception):
    """Raised when the local educational ML artifact is missing or invalid."""


@st.cache_resource
def _load_education_ml_artifact() -> dict:
    """Load and validate the project-local educational classifier artifact."""
    artifact_path = (
        Path(__file__).resolve().parent
        / "models"
        / "education_text_classifier.joblib"
    )
    if not artifact_path.is_file():
        raise MLArtifactError(
            f"The educational model artifact is missing: {artifact_path.name}"
        )

    try:
        payload = joblib.load(artifact_path)
    except Exception as error:
        raise MLArtifactError(
            "The educational model artifact could not be loaded or is corrupt."
        ) from error

    if not isinstance(payload, dict):
        raise MLArtifactError(
            "The educational model artifact is invalid: payload must be a dictionary."
        )

    pipeline = payload.get("pipeline")
    categories = payload.get("categories")
    if (
        not isinstance(categories, (list, tuple))
        or len(categories) != len(SUPPORTED_CATEGORY_LABELS)
        or set(categories) != set(SUPPORTED_CATEGORY_LABELS)
    ):
        raise MLArtifactError(
            "The educational model artifact has invalid category labels."
        )

    dataset_status = payload.get("dataset_status")
    if not isinstance(dataset_status, dict):
        raise MLArtifactError(
            "The educational model artifact has invalid dataset status metadata."
        )
    if (
        dataset_status.get("authorship_status") != "generated"
        or dataset_status.get("review_status") != "unreviewed"
    ):
        raise MLArtifactError(
            "The educational model artifact does not match the required "
            "generated and unreviewed dataset status."
        )

    fit_warning = get_pipeline_fit_warning(pipeline)
    if fit_warning is not None:
        raise MLArtifactError(
            f"The educational model pipeline is invalid or unavailable: "
            f"{fit_warning.message}"
        )

    classifier = pipeline.named_steps["classifier"]
    classifier_classes = tuple(str(category) for category in classifier.classes_)
    supported_categories = set(SUPPORTED_CATEGORY_LABELS)
    class_set = set(classifier_classes)

    if len(classifier_classes) != 5:
        raise MLArtifactError(
            "The educational model artifact is invalid: the classifier "
            "must contain exactly five classes."
        )
    if len(classifier_classes) != len(class_set):
        raise MLArtifactError(
            "The educational model artifact is invalid: the classifier "
            "contains duplicate classes."
        )
    unsupported_classes = sorted(class_set - supported_categories)
    if unsupported_classes:
        raise MLArtifactError(
            "The educational model artifact is invalid: unsupported classifier "
            f"class(es): {', '.join(unsupported_classes)}."
        )
    if class_set != supported_categories:
        raise MLArtifactError(
            "The educational model artifact is invalid: classifier classes "
            "do not exactly match the five supported categories."
        )

    return payload


def render_ml_message(message: str, kind: str) -> None:
    """Render an ML-page-only accessible message with explicit local colors."""
    styles = {
        "warning": {
            "background": "#fef3c7",
            "text": "#713f12",
            "icon": "⚠",
        },
        "error": {
            "background": "#fee2e2",
            "text": "#7f1d1d",
            "icon": "⛔",
        },
        "success": {
            "background": "#dcfce7",
            "text": "#14532d",
            "icon": "✓",
        },
        "info": {
            "background": "#dbeafe",
            "text": "#1e3a8a",
            "icon": "ℹ",
        },
    }
    style = styles.get(kind, styles["info"])
    safe_message = html.escape(str(message))
    safe_icon = html.escape(style["icon"])
    role = "status" if kind in {"success", "info"} else "alert"
    st.markdown(
        f"""
        <div class="ml-message ml-message-{kind}" role="{role}">
            <style>
            .ml-message-{kind} {{
                background-color: {style["background"]} !important;
                color: {style["text"]} !important;
                border: 1px solid {style["text"]};
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin: 0.5rem 0;
            }}
            .ml-message-{kind} *,
            .ml-message-{kind} span,
            .ml-message-{kind} p {{
                color: {style["text"]} !important;
            }}
            </style>
            <span class="ml-message-{kind}-icon">{safe_icon}</span>
            <span class="ml-message-{kind}-text">{safe_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ml_demonstration():
    """Display the isolated educational-text classifier demonstration."""
    st.markdown(
        """
        <div class="section-card">
            <h3>Educational Text Category Demonstration</h3>
            <p>
                This feature uses a TF-IDF and Logistic Regression model to
                categorize educational sentences. It does not diagnose medical
                conditions or interpret blood-test reports.
            </p>
            <p>
                The training dataset contains 25 generated, unreviewed examples.
                This demonstration is not clinically validated or established
                for real-world reliability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        payload = _load_education_ml_artifact()
    except MLArtifactError as error:
        render_ml_message(str(error), "error")
        return
    except Exception:
        render_ml_message(
            "The educational model artifact could not be loaded.",
            "error",
        )
        return

    pipeline = payload["pipeline"]
    sentence = st.text_area(
        "Educational sentence",
        key="ml_education_sentence",
        height=140,
        placeholder="Enter an educational sentence...",
    )

    if st.button("Predict Category", type="primary", use_container_width=True):
        result = predict_education_category(pipeline, sentence)

        for warning in result.warnings:
            if warning.code in {
                "empty_input",
                "invalid_input",
                "short_input",
                "mostly_out_of_vocabulary",
            }:
                render_ml_message(warning.message, "warning")
            else:
                render_ml_message(warning.message, "error")

        if result.predicted_category == "Unavailable":
            render_ml_message("No category is available for this input.", "info")
            return

        st.subheader("Predicted category")
        render_ml_message(result.predicted_category, "success")

        st.caption(
            "These are model output probabilities, not certainty or "
            "calibrated confidence."
        )
        probability_rows = [
            {
                "Category": category,
                "Model output probability": probability,
            }
            for category, probability in result.probabilities.items()
        ]
        st.dataframe(
            probability_rows,
            hide_index=True,
            use_container_width=True,
        )


if selected_page == "Dashboard":
    st.title("Welcome to your health report workspace")
    st.write(
        "Use this educational prototype to explore the planned report "
        "simplification workflow."
    )
    render_report_workspace()
elif selected_page == "Upload Report":
    st.title("Upload Report")
    render_report_workspace()
elif selected_page in ("Analysis Results", "View All Results"):
    st.title(selected_page)
    render_interpretation_table()
elif selected_page == "ML Demonstration":
    st.title(selected_page)
    render_ml_demonstration()
else:
    st.title(selected_page)
    st.markdown(
        """
        <div class="section-card">
            <h3>Coming soon</h3>
            <p>
                This section is a layout placeholder. No medical analysis,
                medicine guidance, or results are available in this prototype.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )