import hashlib
from html import escape as html_escape
import re

import pandas as pd
import plotly.graph_objects as go
import pymupdf
import streamlit as st

from ml_classifier import (
    SYNTHETIC_EXAMPLES,
    get_demo_evaluation,
    predict_category,
)


st.set_page_config(
    page_title="MediExplain",
    page_icon="🩺",
    layout="wide",
)

NUMBER = r"[+-?(]?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)[)]?"

TEST_ALIASES = [
    ("Hemoglobin", [r"hemoglobin", r"\bhgb\b", r"\bhb\b"]),
    ("Red Blood Cells", [r"red blood (?:cell|cells)", r"\brbc\b"]),
    ("White Blood Cells", [r"white blood (?:cell|cells)", r"\bwbc\b"]),
    ("Platelets", [r"platelets?", r"\bplt\b"]),
    ("Hematocrit", [r"hematocrit", r"\bhct\b"]),
    ("MCV", [r"\bmcv\b"]),
    ("MCH", [r"\bmch\b"]),
    ("MCHC", [r"\bmchc\b"]),
    ("RDW", [r"\brdw\b"]),
    ("Neutrophils", [r"neutrophils?"]),
    ("Lymphocytes", [r"lymphocytes?"]),
    ("Glucose", [r"(?:blood )?glucose", r"(?:fasting )?(?:blood )?sugar"]),
    ("HbA1c", [r"\bhba1c\b", r"\bhb a1c\b", r"glycated hemoglobin", r"glycosylated hemoglobin"]),
    ("Total Cholesterol", [r"total cholesterol"]),
    ("LDL Cholesterol", [r"\bldl\b", r"ldl cholesterol"]),
    ("HDL Cholesterol", [r"\bhdl\b", r"hdl cholesterol"]),
    ("Triglycerides", [r"triglycerides?"]),
    ("Creatinine", [r"creatinine"]),
    ("Blood Urea Nitrogen", [r"blood urea nitrogen", r"\bbun\b"]),
    ("Urea", [r"\burea\b"]),
    ("Uric Acid", [r"uric acid"]),
    ("AST", [r"\bast\b", r"aspartate aminotransferase"]),
    ("ALT", [r"\balt\b", r"alanine aminotransferase"]),
    ("ALP", [r"\balp\b", r"alkaline phosphatase"]),
    ("Bilirubin", [r"(?:total )?bilirubin"]),
    ("TSH", [r"\btsh\b", r"thyroid stimulating hormone"]),
    ("T3", [r"\bt3\b", r"tri-iodothyronine"]),
    ("T4", [r"\bt4\b", r"thyroxine"]),
    ("Vitamin D", [r"vitamin d", r"25-oh vitamin d", r"25-oh d"]),
    ("Vitamin B12", [r"vitamin b12", r"\bb12\b"]),
]

TEST_MEASURES = {
    "Hemoglobin": "Hemoglobin is the oxygen-carrying protein in red blood cells.",
    "Red Blood Cells": "This count measures the number of red blood cells in a blood sample.",
    "White Blood Cells": "This count measures white blood cells, which are part of the immune system.",
    "Platelets": "Platelets are blood components involved in clot formation.",
    "Hematocrit": "Hematocrit is the proportion of blood volume made up of red blood cells.",
    "MCV": "Mean corpuscular volume describes the average size of red blood cells.",
    "MCH": "Mean corpuscular hemoglobin describes the average hemoglobin amount per red blood cell.",
    "MCHC": "Mean corpuscular hemoglobin concentration describes hemoglobin concentration within red blood cells.",
    "RDW": "Red cell distribution width describes variation in red blood cell size.",
    "Neutrophils": "Neutrophils are a type of white blood cell involved in immune responses.",
    "Lymphocytes": "Lymphocytes are white blood cells involved in immune responses.",
    "Glucose": "This test measures glucose, a form of sugar, in the blood; fasting status and timing can matter.",
    "HbA1c": "HbA1c measures hemoglobin with glucose attached and reflects average blood glucose over roughly the prior 2–3 months.",
    "Total Cholesterol": "This measures the total amount of cholesterol carried in the blood.",
    "LDL Cholesterol": "LDL cholesterol is one type of cholesterol-carrying particle measured in the blood.",
    "HDL Cholesterol": "HDL cholesterol is one type of cholesterol-carrying particle measured in the blood.",
    "Triglycerides": "Triglycerides are a type of fat measured in the blood.",
    "Creatinine": "Creatinine is a waste product from normal muscle activity; interpretation may use other kidney tests and context.",
    "Blood Urea Nitrogen": "This measures nitrogen in urea, a waste product processed by the kidneys.",
    "Urea": "Urea is a waste product formed during protein breakdown and removed from the blood by the kidneys.",
    "Uric Acid": "Uric acid is produced when the body breaks down purines from cells and some foods.",
    "AST": "AST is an enzyme found in the liver and other tissues; this result is interpreted with other tests and context.",
    "ALT": "ALT is an enzyme found mostly in the liver; this result is interpreted with other tests and context.",
    "ALP": "Alkaline phosphatase is an enzyme found in the liver, bile ducts, and bone.",
    "Bilirubin": "Bilirubin is a yellow breakdown product of red blood cells that the liver processes.",
    "TSH": "TSH is a pituitary hormone that signals the thyroid gland.",
    "T3": "T3 is a thyroid hormone involved in regulating the body's use of energy.",
    "T4": "T4 is a thyroid hormone involved in regulating the body's use of energy.",
    "Vitamin D": "This test commonly measures a circulating form of vitamin D used to assess vitamin D status.",
    "Vitamin B12": "This test measures vitamin B12, a nutrient needed for blood-cell formation and nerve function.",
}


def clean_number(raw):
    if raw is None:
        return None
    value = str(raw).strip().replace(",", "").strip("<>= ")
    value = value.replace("(", "").replace(")", "")
    try:
        return float(value)
    except ValueError:
        return None


def _find_range(line):
    bounded = re.search(
        r"(\d+(?:,\d{3})*(?:\.\d+)?\s*(?:-|–|—|to)\s*\d+(?:,\d{3})*(?:\.\d+)?)",
        line,
        re.I,
    )
    if bounded:
        return "bounded", bounded

    upper = re.search(
        r"(<=?|less than|up to)\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        line,
        re.I,
    )
    if upper:
        return "upper", upper

    lower = re.search(
        r"(>=?|greater than|at least)\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        line,
        re.I,
    )
    if lower:
        return "lower", lower
    return None, None


def extract_range(line):
    range_type, match = _find_range(line)
    if match is None:
        return ""
    if range_type == "bounded":
        return match.group(1).replace(" to ", " - ").strip()
    symbol = match.group(1).strip()
    if range_type == "upper" and ("less" in symbol.lower() or "up" in symbol.lower()):
        symbol = "<="
    elif range_type == "lower" and ("greater" in symbol.lower() or "at least" in symbol.lower()):
        symbol = ">="
    return f"{symbol}{match.group(2)}"


def extract_unit(line, value_match):
    remainder = line[value_match.end():]
    _, range_match = _find_range(remainder)
    if range_match:
        remainder = remainder[:range_match.start()]
    remainder = re.sub(r"\bunits?\s*[:=]?", " ", remainder, flags=re.I)
    remainder = re.sub(r"[\r\n|;]+", " ", remainder)
    remainder = re.sub(r"[\[\](){}]", " ", remainder)
    unit_match = re.search(
        r"^\s*(?:([x×]\s*10\^?\s*\d+\s*/\s*[A-Za-zµμ]+)|([A-Za-zµμ%]+(?:\s*/\s*[A-Za-zµμ%]+(?:³|3|2)?)?))",
        remainder,
    )
    if not unit_match:
        return ""
    unit = unit_match.group(1) or unit_match.group(2)
    return re.sub(r"\s+", "", unit).replace("^", "^").strip()


def parse_report(text):
    records = []
    test_matches = []
    for canonical, aliases in TEST_ALIASES:
        alias_pattern = "|".join(f"(?:{alias})" for alias in aliases)
        test_re = re.compile(alias_pattern, re.I)
        test_matches.extend(
            (match.start(), match.end(), canonical)
            for match in test_re.finditer(text)
        )

    test_matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    unique_matches = []
    for match in test_matches:
        if unique_matches and match[0] < unique_matches[-1][1]:
            continue
        unique_matches.append(match)

    for index, (_, end, canonical) in enumerate(unique_matches):
        next_start = unique_matches[index + 1][0] if index + 1 < len(unique_matches) else len(text)
        section = text[end:next_start]

        _, range_match = _find_range(section)
        ref_range = extract_range(section)
        value_section = section
        if range_match:
            value_section = section[:range_match.start()] + " " * (range_match.end() - range_match.start()) + section[range_match.end():]

        value_match = re.search(
            r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z])",
            value_section,
        )
        if value_match is None:
            continue

        value = clean_number(value_match.group(0))
        if value is None:
            continue

        records.append({
            "Test Name": canonical,
            "Value": value,
            "Unit": extract_unit(section, value_match),
            "Reference Range": ref_range,
        })

    return pd.DataFrame(records, columns=["Test Name", "Value", "Unit", "Reference Range"])


def classify_value(value, range_text):
    if value is None or pd.isna(value) or not range_text:
        return "Cannot Assess"

    range_text = str(range_text).strip()

    bounded = re.match(
        r"^\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        range_text,
        re.I,
    )
    if bounded:
        low = clean_number(bounded.group(1))
        high = clean_number(bounded.group(2))
        if low is None or high is None:
            return "Cannot Assess"
        if value < low:
            return "Low"
        if value > high:
            return "High"
        return "Within Range"

    upper = re.match(r"^\s*(<=?|<)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*$", range_text)
    if upper:
        limit = clean_number(upper.group(2))
        if limit is None:
            return "Cannot Assess"
        return "Within Range" if value <= limit else "High"

    lower = re.match(r"^\s*(>=?|>)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*$", range_text)
    if lower:
        limit = clean_number(lower.group(2))
        if limit is None:
            return "Cannot Assess"
        return "Within Range" if value >= limit else "Low"

    return "Cannot Assess"


def result_guidance(test_name, status, language):
    is_hinglish = language == "Hinglish"
    if is_hinglish:
        direction = "upar" if status == "High" else "neeche"
        explanation = (
            f"{test_name} ki value is report ke apne reference range se {direction} hai. "
            "Yeh akela result kisi condition ka diagnosis nahi karta; clinical context zaroori hai."
        )
    else:
        direction = (
            "above its own report-specific reference range"
            if status == "High"
            else "below its own report-specific reference range"
        )
        explanation = (
            f"The {test_name} value is {direction}. "
            "This result alone does not diagnose a condition; clinical context matters."
        )

    if test_name in {"Hemoglobin", "Red Blood Cells", "Hematocrit", "MCV", "MCH", "MCHC", "RDW"}:
        content = {
            "English": (
                "Choose varied meals such as dal, chana, rajma, leafy greens, whole grains, and fruit. Food choices do not replace evaluation or prescribed care.",
                "Keep to comfortable, regular movement; reduce intensity and seek advice if you feel unwell during activity.",
                "Ask whether the full blood count, symptoms, diet, or other context changes how this result should be read. Avoid starting supplements on your own.",
                "Could this result be interpreted with the other blood-count values and my health context?",
            ),
            "Hinglish": (
                "Dal, chana, rajma, hari sabzi, whole grains aur phal jaise varied foods lein. Khana clinical check-up ya prescribed care ka replacement nahi hai.",
                "Apni sahulat ke mutabik regular movement karein; tabiyat kharab lage to activity kam karke salah lein.",
                "Doctor se poochhein ki poori blood-count report, symptoms, diet ya doosra context is result ko kaise affect karta hai. Apne aap supplements shuru na karein.",
                "Kya is result ko baaki blood-count values aur meri health history ke saath dekhna chahiye?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"White Blood Cells", "Neutrophils", "Lymphocytes", "Platelets"}:
        content = {
            "English": (
                "Aim for regular balanced meals with dal/beans, vegetables, fruit, and whole grains; no single food reliably corrects a blood-count result.",
                "Continue usual activity if you feel well; pause strenuous activity and seek clinical advice if you feel unwell.",
                "Review this alongside symptoms and the rest of the blood count. Do not use foods or supplements as a substitute for follow-up.",
                "Does this need a repeat test or interpretation alongside my symptoms and other blood-count results?",
            ),
            "Hinglish": (
                "Dal/beans, sabzi, phal aur whole grains ke saath balanced meals rakhein; koi ek food blood-count result ko reliably theek nahi karta.",
                "Agar aap theek mehsoos kar rahe hain to normal activity jaari rakhein; tabiyat kharab ho to strenuous activity rok kar salah lein.",
                "Is result ko symptoms aur poori blood-count report ke saath samjhein. Follow-up ki jagah foods ya supplements par nirbhar na karein.",
                "Kya symptoms aur doosre blood-count results ke saath repeat test ya review ki zaroorat hai?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"Glucose"}:
        content = {
            "English": (
                "Consider balanced portions with vegetables, dal/beans, and whole grains such as roti or brown rice; limit sugary drinks. Whether the test was fasting matters.",
                "If appropriate for you, regular walking or other comfortable movement can support general wellbeing; avoid sudden intense changes.",
                "Confirm whether the sample was fasting and review this with your clinician; do not change medicines based on this app.",
                "Was this test fasting, and should it be reviewed with another glucose test or my symptoms?",
            ),
            "Hinglish": (
                "Balanced portions mein sabzi, dal/beans aur whole grains jaise roti ya brown rice rakhein; meethi drinks kam karein. Test fasting tha ya nahi, yeh maayne rakhta hai.",
                "Agar aapke liye theek ho, regular walk ya comfortable movement karein; achanak bahut intense exercise shuru na karein.",
                "Sample fasting tha ya nahi, doctor se confirm karein. Is app ke basis par medicines change na karein.",
                "Kya test fasting tha, aur kya ise kisi doosre glucose test ya symptoms ke saath dekhna chahiye?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides"}:
        content = {
            "English": (
                "Favor vegetables, fruit, dal/beans, whole grains, and nuts in suitable portions; choose less fried food and fewer trans-fat-rich packaged snacks.",
                "Build toward regular moderate activity that suits your abilities; start gradually if you are currently inactive.",
                "Discuss the complete lipid profile and personal risk factors with your clinician. Avoid interpreting one cholesterol number in isolation.",
                "How should this value be interpreted with my full lipid profile and personal risk factors?",
            ),
            "Hinglish": (
                "Sabzi, phal, dal/beans, whole grains aur munasib portions mein nuts lein; fried food aur trans-fat wale packaged snacks kam rakhein.",
                "Apni capacity ke mutabik regular moderate activity karein; abhi inactive hain to dheere-dheere shuru karein.",
                "Poora lipid profile aur personal risk factors doctor se discuss karein. Ek cholesterol number ko akela interpret na karein.",
                "Mere poore lipid profile aur personal risk factors ke saath is value ka kya matlab hai?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"Creatinine", "Blood Urea Nitrogen", "Urea", "Uric Acid"}:
        content = {
            "English": (
                "Choose a balanced diet with familiar foods such as dal, vegetables, fruit, and grains. Do not make major protein or fluid changes without clinician advice.",
                "Continue comfortable activity if you feel well; ask your clinician about limits if you have symptoms or existing health concerns.",
                "Interpret with other results, medicines/supplements, and health history. Do not start a restrictive diet or change fluid intake based on one value.",
                "Could medicines, supplements, recent activity, or other results affect this value, and is follow-up needed?",
            ),
            "Hinglish": (
                "Dal, sabzi, phal aur grains jaise familiar foods ke saath balanced diet rakhein. Doctor ki salah ke bina protein ya fluids mein bade changes na karein.",
                "Agar aap theek mehsoos kar rahe hain to comfortable activity jaari rakhein; symptoms ya health concerns hon to limits poochhein.",
                "Is result ko doosre tests, medicines/supplements aur health history ke saath samjhein. Ek value dekh kar restrictive diet ya fluid change na karein.",
                "Kya medicines, supplements, recent activity ya doosre results is value ko affect kar sakte hain, aur follow-up chahiye?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"AST", "ALT", "ALP", "Bilirubin"}:
        content = {
            "English": (
                "Choose varied meals with vegetables, fruit, dal/beans, and whole grains; avoid relying on 'detox' foods or supplements.",
                "Regular comfortable activity is reasonable if you feel well; avoid sudden strenuous changes.",
                "Review the full panel, symptoms, alcohol use, and medicines/supplements with a clinician. Do not stop prescribed treatment yourself.",
                "Should this be interpreted with the other liver-panel results, symptoms, or my medicines and supplements?",
            ),
            "Hinglish": (
                "Sabzi, phal, dal/beans aur whole grains ke varied meals lein; 'detox' foods ya supplements par bharosa na karein.",
                "Agar aap theek hain to comfortable regular activity karein; achanak strenuous routine na shuru karein.",
                "Poora panel, symptoms, alcohol use aur medicines/supplements doctor se discuss karein. Prescribed treatment apne aap band na karein.",
                "Kya ise baaki liver-panel results, symptoms, ya meri medicines aur supplements ke saath dekhna chahiye?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"TSH", "T3", "T4"}:
        content = {
            "English": (
                "Maintain a varied diet with ordinary sources such as dairy or fortified alternatives, pulses, and vegetables; avoid high-dose iodine or 'thyroid' supplements unless advised.",
                "Keep a regular activity routine that feels comfortable; activity does not replace evaluation of this result.",
                "Timing, symptoms, and medicines/supplements can matter. Do not start, stop, or adjust treatment based on this app.",
                "Should this be reviewed with other thyroid tests, symptoms, or the timing of my medicines and supplements?",
            ),
            "Hinglish": (
                "Dairy ya fortified alternatives, pulses aur sabzi jaise normal foods ke saath varied diet rakhein; salah ke bina high-dose iodine ya 'thyroid' supplements na lein.",
                "Apni sahulat ke mutabik regular activity karein; exercise is result ke medical review ka replacement nahi hai.",
                "Test timing, symptoms aur medicines/supplements maayne rakh sakte hain. Is app ke basis par treatment shuru, band ya adjust na karein.",
                "Kya ise doosre thyroid tests, symptoms, ya medicines/supplements ke timing ke saath review karna chahiye?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    elif test_name in {"Vitamin D", "Vitamin B12"}:
        content = {
            "English": (
                "Include suitable food sources such as dairy/fortified alternatives, eggs or fish if part of your diet, and varied vegetables; food alone may not explain the result.",
                "Choose regular movement that suits you; do not use sun exposure as a treatment plan.",
                "Discuss causes and next steps with your clinician. Do not begin high-dose supplements or change an existing dose without their advice.",
                "Could diet, symptoms, or other tests help explain this result, and should I discuss supplementation?",
            ),
            "Hinglish": (
                "Dairy/fortified alternatives, diet mein ho to eggs ya fish, aur varied sabzi shamil karein; sirf diet se result ka reason pata nahi chalta.",
                "Apni sahulat ke mutabik regular movement karein; dhoop ko treatment plan na samjhein.",
                "Possible reasons aur next steps doctor se poochhein. Salah ke bina high-dose supplements shuru ya existing dose change na karein.",
                "Kya diet, symptoms ya doosre tests is result ko samajhne mein madad karenge, aur supplementation par baat karni chahiye?",
            ),
        }["Hinglish" if is_hinglish else "English"]
    else:
        content = (
            (
                "Keep a varied, balanced diet using familiar foods such as dal/beans, vegetables, fruit, and whole grains.",
                "Choose regular, comfortable movement suited to your abilities.",
                "This value needs interpretation in context. Avoid self-treatment and discuss persistent or concerning results with a clinician.",
                "What context or other test results should be considered when interpreting this value?",
            )
            if not is_hinglish
            else (
                "Dal/beans, sabzi, phal aur whole grains ke saath varied, balanced diet rakhein.",
                "Apni capacity ke mutabik comfortable regular movement karein.",
                "Is value ko health context mein samajhna zaroori hai. Self-treatment na karein aur concern ho to clinician se baat karein.",
                "Is value ko samajhne ke liye kaunsa context ya doosra test dekhna chahiye?",
            )
        )

    return {
        "explanation": explanation,
        "measurement": TEST_MEASURES.get(
            test_name,
            "A test-specific explanation is unavailable for this name; ask a qualified clinician what it measures.",
        ),
        "diet": content[0],
        "activity": content[1],
        "lifestyle": content[2],
        "doctor_question": content[3],
    }


def render_section_heading(anchor, icon, title, subtitle):
    st.markdown(
        f"""
        <section id="nav-{html_escape(anchor)}" class="section-heading">
          <div class="section-heading-icon">{html_escape(icon)}</div>
          <div><h2>{html_escape(title)}</h2><p>{html_escape(subtitle)}</p></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(report_available, summary_available):
    items = [("upload-laboratory-report", "📄", "Upload Laboratory Report")]
    if report_available:
        items.append(("review-extracted-values", "🧾", "Review Extracted Values"))
    if summary_available:
        items.extend([
            ("overall-report-summary", "📊", "Overall Report Summary"),
            ("results-that-need-attention", "🔎", "Results That Need Attention"),
            ("diet-lifestyle-guidance", "🥗", "Diet and Lifestyle Guidance"),
            ("medication-ingredient-information", "💊", "Medication Ingredient Information"),
            ("questions-for-your-doctor", "❔", "Questions for Your Doctor"),
            ("complete-test-results", "📋", "Complete Test Results"),
            ("download-summary", "⬇️", "Download Summary"),
        ])
    links = "".join(
        f'<a href="#nav-{anchor}"><span>{icon}</span>{html_escape(title)}</a>'
        for anchor, icon, title in items
    )
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand"><strong>MediExplain</strong><span>Clear, report-based health information</span></div>
        <div class="sidebar-nav-title">Navigate this report</div>
        <nav class="sidebar-nav">{links}</nav>
        <div class="sidebar-about"><strong>About MediExplain</strong>
          Educational information only. This app does not diagnose conditions or recommend medicines.
          Review results with a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


MEDICATION_INGREDIENT_INFO = [
    {
        "ingredient": "Iron salts (for example, ferrous sulfate)",
        "purpose": "Provide iron used by the body to make hemoglobin and red blood cells.",
        "condition": "A clinician-confirmed iron deficiency; the app does not infer this from hemoglobin or other blood-count values.",
        "evaluation": "Low hemoglobin can have many causes. A clinician may review iron studies, blood counts, symptoms, medical history, and possible sources of blood loss or reduced absorption.",
        "precautions": "Iron may cause stomach upset or constipation, can interact with some medicines, and excess iron can be harmful. Accidental overdose is especially dangerous to children; keep iron products securely stored.",
        "reference": "https://medlineplus.gov/druginfo/meds/a682778.html",
        "reference_label": "MedlinePlus: Iron Supplements",
    },
    {
        "ingredient": "Vitamin B12 (cyanocobalamin)",
        "purpose": "Replaces vitamin B12 when deficiency is confirmed; the appropriate form and route depend on the cause and absorption.",
        "condition": "A clinician-confirmed vitamin B12 deficiency, including deficiency related to impaired absorption in some cases.",
        "evaluation": "A low result needs clinical interpretation and assessment of possible causes, including diet, absorption, health conditions, and medicines.",
        "precautions": "A clinician should review allergies (including cobalt), kidney disease, other conditions, and current medicines. Injectable cyanocobalamin is administered under clinical guidance.",
        "reference": "https://medlineplus.gov/druginfo/meds/a605007.html",
        "reference_label": "MedlinePlus: Cyanocobalamin Injection",
    },
]


DEMO_TEXT = """SYNTHETIC LAB REPORT - FOR DEMONSTRATION ONLY
Hemoglobin 13.8 g/dL 12.0 - 16.0
White Blood Cells 7.2 x10^3/uL 4.0 - 11.0
Platelets 285 x10^3/uL 150 - 450
Glucose 92 mg/dL 70 - 100
LDL Cholesterol 176 mg/dL 0 - 130
Creatinine 1.6 mg/dL 0.6 - 1.2
"""

for state_key, default in {
    "report_text": "",
    "source_name": "",
    "input_signature": "",
    "uploaded_signature": "",
    "reviewed_rows": [],
    "summary_visible": False,
    "summary_rows": [],
    "parsing_warning": False,
    "demo_load_count": 0,
    "demo_loaded": False,
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default


def load_demo_report():
    st.session_state.demo_load_count += 1
    st.session_state.report_text = DEMO_TEXT
    st.session_state.source_name = "Synthetic Demo Report"
    st.session_state.input_signature = f"synthetic-demo-{st.session_state.demo_load_count}"
    st.session_state.reviewed_rows = parse_report(DEMO_TEXT).to_dict("records")
    st.session_state.parsing_warning = not bool(st.session_state.reviewed_rows)
    st.session_state.summary_visible = False
    st.session_state.summary_rows = []
    st.session_state.demo_loaded = True


st.markdown(
    """
    <style>
        :root { --ink:#202447; --muted:#707897; --line:#e9e9f5; --violet:#7557d9; --blue:#4776e6; --teal:#20b9a7; --coral:#f27d78; }
        .stApp { background:radial-gradient(ellipse at 12% 8%,rgba(196,174,255,.25),transparent 30rem),radial-gradient(ellipse at 92% 28%,rgba(82,211,197,.16),transparent 28rem),linear-gradient(180deg,#f9f8ff 0%,#f5f8ff 52%,#f9fbff 100%); color:var(--ink); }
        .block-container { max-width:1240px; padding-top:.35rem; padding-bottom:3rem; }
        #MainMenu, footer { visibility:hidden; }
        header { background:transparent !important; }
        .brandbar { display:flex;align-items:center;gap:11px;width:fit-content;margin:0 0 14px;padding:6px 13px 6px 7px;border:1px solid rgba(255,255,255,.88);border-radius:17px;background:rgba(255,255,255,.78);box-shadow:0 8px 20px rgba(72,73,137,.08),inset 0 1px 0 #fff;animation:fadeUp .42s ease-out both; }
        .brand-logo { position:relative;isolation:isolate;width:42px;height:42px;flex:0 0 42px;border:1px solid rgba(255,255,255,.65);border-radius:14px;background:linear-gradient(140deg,#7654d4 0%,#4f78e6 58%,#24b9aa 100%);box-shadow:0 7px 15px rgba(83,77,174,.22),inset 0 1px 0 rgba(255,255,255,.42);transform:rotate(-5deg); }
        .brand-logo-m,.brand-logo-e { position:absolute;font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-weight:900;line-height:1;text-shadow:0 2px 6px rgba(31,35,100,.2); }
        .brand-logo-m { left:5px;top:8px;color:#fff;font-size:27px; }
        .brand-logo-e { right:5px;top:13px;color:#a8ffed;font-size:23px;transform:rotate(5deg); }
        .brand-logo-pill { position:absolute;right:6px;bottom:5px;width:14px;height:7px;border:1px solid rgba(255,255,255,.92);border-radius:999px;background:linear-gradient(90deg,#ff92b7 0 50%,#fff 50%);transform:rotate(-38deg);box-shadow:0 1px 5px rgba(32,38,113,.25); }
        .brand-name { color:#292b52;font-size:1.08rem;font-weight:850;letter-spacing:-.035em; }
        .upload-section-anchor { height:1px;scroll-margin-top:1rem; }
        .st-key-upload_workspace { margin-bottom:20px; }
        .st-key-upload_workspace [data-testid="stColumn"] { min-width:0; padding:23px 24px 20px; border:1px solid rgba(255,255,255,.86); border-radius:25px; background:linear-gradient(145deg,rgba(255,255,255,.92),rgba(249,249,255,.84)); box-shadow:0 16px 42px rgba(73,72,137,.09),inset 0 1px 0 #fff; transition:transform .28s ease,box-shadow .28s ease; transform-style:preserve-3d; animation:fadeUp .65s ease-out backwards; }
        .st-key-upload_workspace [data-testid="stColumn"]:hover { transform:translateY(-4px) rotateX(.6deg);box-shadow:0 24px 48px rgba(73,72,137,.14),inset 0 1px 0 #fff; }
        .st-key-upload_workspace [data-testid="stColumn"]:first-child { animation-delay:.08s; }
        .st-key-upload_workspace [data-testid="stColumn"]:last-child { animation-delay:.17s; }
        .card { border:0; background:transparent; }
        .card-title { display:flex; align-items:center; gap:11px; color:var(--ink); font-size:1.14rem; font-weight:800; letter-spacing:-.02em; margin:0 0 13px; }
        .card-title span { display:grid; place-items:center; width:40px;height:40px;border:1px solid rgba(117,87,217,.12);border-radius:14px;background:linear-gradient(145deg,#f1eaff,#e6fbf7);box-shadow:0 7px 15px rgba(101,83,187,.1); }
        .upload-visual { display:grid;grid-template-columns:minmax(115px,.72fr) minmax(0,1.3fr);align-items:center;gap:18px;min-height:145px;margin:2px 0 12px;padding:13px 15px;border:1px solid rgba(228,225,249,.9);border-radius:21px;background:radial-gradient(circle at 10% 12%,rgba(208,191,255,.3),transparent 43%),linear-gradient(135deg,#fbfaff,#f3fbff); }
        .doc-scene { position:relative;height:118px;display:grid;place-items:center;perspective:650px; }
        .doc-shadow { position:absolute;width:82px;height:99px;border-radius:14px;background:linear-gradient(155deg,#bca9fb,#889cf4);transform:translate(10px,6px) rotate(9deg);box-shadow:0 12px 22px rgba(91,93,183,.2); }
        .doc-sheet { position:relative;width:76px;height:98px;padding:13px 10px;border:1px solid rgba(255,255,255,.8);border-radius:13px;background:linear-gradient(150deg,#fff,#f8fbff);box-shadow:0 14px 25px rgba(62,76,151,.17);transform:rotateY(-12deg) rotateX(4deg);animation:docHover 5s ease-in-out infinite; }
        .doc-sheet:before { content:"";position:absolute;right:0;top:0;width:22px;height:22px;border-radius:0 12px 0 8px;background:linear-gradient(135deg,#e8e0ff,#d6f7f1); }
        .doc-cross { width:22px;height:22px;margin-bottom:12px;display:grid;place-items:center;border-radius:7px;background:linear-gradient(140deg,#7657d9,#4a8ce9);color:#fff;font-size:15px;font-weight:800; }
        .doc-line { height:4px;margin:6px 0;border-radius:5px;background:#dfe4f4; }
        .doc-line.short { width:58%;background:#cbbaff; }
        .doc-heart { position:absolute;right:3px;top:8px;display:grid;place-items:center;width:34px;height:34px;border-radius:12px;background:linear-gradient(145deg,#fff,#fff1f4);color:#ef7b9b;box-shadow:0 8px 17px rgba(218,105,144,.2);animation:floatB 4.8s ease-in-out infinite; }
        .doc-plus { position:absolute;left:7px;bottom:9px;display:grid;place-items:center;width:27px;height:27px;border-radius:10px;background:linear-gradient(145deg,#dbfff6,#edfffb);color:#17aa9e;box-shadow:0 8px 17px rgba(32,185,167,.16); }
        .upload-copy strong { display:block;color:#292b52;font-size:.95rem;font-weight:800;letter-spacing:-.015em; }
        .upload-copy p { margin:5px 0 9px;color:#707897;font-size:.78rem;line-height:1.45; }
        .format-chips { display:flex;flex-wrap:wrap;gap:6px; }
        .format-chip { padding:4px 9px;border:1px solid #e5e2f6;border-radius:999px;background:rgba(255,255,255,.76);color:#6550b8;font-size:.66rem;font-weight:800;letter-spacing:.04em; }
        .workflow { display:flex;align-items:center;justify-content:space-between;gap:5px;margin-top:12px;color:#646b8a;font-size:.68rem;font-weight:750; }
        .workflow span { display:flex;align-items:center;gap:5px;white-space:nowrap; }
        .workflow b { display:grid;place-items:center;width:20px;height:20px;border-radius:7px;background:linear-gradient(135deg,#eee9ff,#e2fbf6);color:#684fca;font-size:.65rem; }
        .workflow i { flex:1;height:1px;min-width:8px;background:linear-gradient(90deg,#c8b9f4,#9fe6d9); }
        [data-testid="stFileUploader"] { margin-top:4px;border:1.5px dashed #c8c0eb;border-radius:16px;background:linear-gradient(145deg,rgba(255,255,255,.94),rgba(247,249,255,.92));padding:10px 12px;transition:border-color .2s ease,background .2s ease,box-shadow .2s ease; }
        [data-testid="stFileUploader"]:hover { border-color:#8c72e8;background:#fbf9ff;box-shadow:0 0 0 4px rgba(117,87,217,.07); }
        [data-testid="stFileUploader"] section { border:0;background:transparent; }
        .demo-scene { position:relative;display:grid;grid-template-columns:112px 1fr;align-items:center;gap:10px;min-height:116px;margin:2px 0 10px;padding:9px 11px;border:1px solid rgba(226,224,246,.92);border-radius:20px;background:radial-gradient(circle at 20% 45%,rgba(191,224,255,.37),transparent 42%),linear-gradient(135deg,#fbf8ff,#f0fcfa);overflow:hidden; }
        .lab-illustration { position:relative;height:94px;display:grid;place-items:center; }
        .lab-glow { position:absolute;width:87px;height:87px;border-radius:50%;background:radial-gradient(circle,rgba(142,117,237,.18),rgba(49,196,177,.08) 55%,transparent 70%); }
        .flask { position:relative;width:52px;height:60px;margin-top:20px;border:3px solid #7885df;border-top:0;border-radius:10px 10px 18px 18px;background:linear-gradient(180deg,rgba(255,255,255,.7) 20%,rgba(104,217,197,.68) 21%,rgba(75,177,226,.74));box-shadow:0 9px 18px rgba(89,102,183,.2),inset 0 0 12px rgba(255,255,255,.65);transform:rotate(-8deg);animation:flaskBob 4.6s ease-in-out infinite; }
        .flask:before { content:"";position:absolute;left:13px;top:-20px;width:20px;height:22px;border:3px solid #7885df;border-bottom:0;border-radius:5px 5px 0 0;background:linear-gradient(90deg,rgba(255,255,255,.4),rgba(232,228,255,.45)); }
        .flask:after { content:"";position:absolute;left:-5px;top:28px;width:56px;height:9px;border-top:2px solid rgba(255,255,255,.85);border-radius:50%;transform:rotate(-4deg); }
        .flask-bubble { position:absolute;z-index:1;width:5px;height:5px;border-radius:50%;background:#f4ffff;box-shadow:0 0 7px rgba(255,255,255,.8);animation:bubbleRise 3.4s ease-in infinite; }
        .bubble-one { left:44px;top:51px; }.bubble-two { left:61px;top:59px;animation-delay:1.2s;width:4px;height:4px; }.bubble-three { left:51px;top:71px;animation-delay:2s;width:3px;height:3px; }
        .lab-spark { position:absolute;z-index:2;color:#ee9e77;font-size:1rem;animation:sparkle 3s ease-in-out infinite; }.spark-one{left:12px;top:17px}.spark-two{right:7px;bottom:17px;animation-delay:1.1s;color:#7c66d8}
        .lab-illustration,.lab-glow,.flask,.flask-bubble,.lab-spark { pointer-events:none; }
        .demo-stats { display:grid;grid-template-columns:1fr;gap:6px; }
        .demo-stat { display:flex;align-items:center;justify-content:space-between;gap:7px;padding:6px 9px;border:1px solid rgba(255,255,255,.9);border-radius:11px;background:rgba(255,255,255,.78);box-shadow:0 4px 12px rgba(72,79,142,.06);color:#626b8a;font-size:.68rem;font-weight:700;transition:transform .2s ease,box-shadow .2s ease; }
        .demo-stat:hover { transform:translateX(3px) scale(1.02);box-shadow:0 7px 14px rgba(72,79,142,.12); }
        .demo-stat strong { color:#373963;font-size:.82rem; }
        .demo-stat .mint { color:#0e9d8a; }.demo-stat .coral{color:#e56f66}
        .sample-note { margin:0 0 9px;color:#858ba3;font-size:.72rem;line-height:1.4; }
        .stButton > button { min-height:44px;border:1px solid rgba(255,255,255,.28);border-radius:14px;padding:.68rem 1.1rem;color:white;background:linear-gradient(105deg,#7754d7 0%,#5f6be3 52%,#438de2 100%);box-shadow:0 10px 22px rgba(89,82,196,.23),inset 0 1px 0 rgba(255,255,255,.24);font-weight:750;letter-spacing:.01em;transition:transform .2s ease,box-shadow .2s ease,filter .2s ease; }
        .stButton > button:hover { transform:translateY(-2px) scale(1.012);box-shadow:0 15px 28px rgba(89,82,196,.3),inset 0 1px 0 rgba(255,255,255,.28);color:white;filter:saturate(1.08); }
        .stButton > button:active { transform:translateY(1px) scale(.99);box-shadow:0 5px 12px rgba(89,82,196,.2); }
        .stButton > button:focus-visible { outline:3px solid rgba(95,107,227,.28);outline-offset:3px; }
        .st-key-upload_workspace [data-testid="stColumn"]:last-child .stButton { position:relative;z-index:3; }
        .report-panel { position:relative;margin-top:22px;border-radius:25px;padding:clamp(18px,3vw,30px);border:1px solid rgba(255,255,255,.9);background:rgba(255,255,255,.82);backdrop-filter:blur(16px);box-shadow:0 18px 52px rgba(64,68,129,.1),inset 0 1px 0 #fff;animation:fadeUp .55s ease-out both; }
        .report-panel h2,.report-panel h3 { color:#292b52;letter-spacing:-.025em; }
        .report-panel [data-testid="stTextArea"] textarea { border-color:#dfdef1;border-radius:13px;background:#fbfbff; }
        .report-panel [data-testid="stDataEditor"] { overflow:hidden;border:1px solid #e5e3f3;border-radius:16px;background:white; }
        .summary-grid { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:16px 0 22px; }
        .summary-card { position:relative;isolation:isolate;overflow:hidden;display:grid;grid-template-columns:50px 1fr;grid-template-rows:auto auto;align-content:center;column-gap:13px;min-height:124px;padding:19px;border:1px solid rgba(255,255,255,.44);border-radius:21px;color:white;box-shadow:0 13px 25px rgba(61,74,130,.14),inset 0 1px 0 rgba(255,255,255,.24);transition:transform .25s ease,box-shadow .25s ease;animation:metricEnter .55s cubic-bezier(.2,.8,.2,1) backwards; }
        .summary-card:after { content:"";position:absolute;z-index:-1;width:130px;height:130px;right:-48px;top:-58px;border:1px solid rgba(255,255,255,.18);border-radius:50%;box-shadow:0 0 0 16px rgba(255,255,255,.04),0 0 0 32px rgba(255,255,255,.025); }
        .summary-card:hover { transform:translateY(-5px) rotateX(2deg) scale(1.015);box-shadow:0 20px 34px rgba(61,74,130,.2); }
        .summary-card:nth-child(2){animation-delay:.06s}.summary-card:nth-child(3){animation-delay:.12s}.summary-card:nth-child(4){animation-delay:.18s}
        .metric-icon { grid-row:1 / span 2;display:grid;place-items:center;width:48px;height:48px;border:1px solid rgba(255,255,255,.3);border-radius:16px;background:rgba(255,255,255,.17);backdrop-filter:blur(8px);font-size:1.35rem;box-shadow:inset 0 1px 0 rgba(255,255,255,.25); }
        .summary-card strong { align-self:end;display:block;font-size:2.1rem;line-height:.96;letter-spacing:-.05em; }
        .summary-card span { align-self:start;display:block;margin-top:7px;font-size:.8rem;font-weight:750;letter-spacing:.01em;color:rgba(255,255,255,.93); }
        .within { background:linear-gradient(135deg,#18a995,#51c9ad); }.high { background:linear-gradient(135deg,#ed6c68,#f6a06e); }.low { background:linear-gradient(135deg,#4776e6,#7a91ed); }.unknown { background:linear-gradient(135deg,#737da5,#929fc0); }
        .report-panel [data-testid="stAlert"] { border:1px solid #e4e6f4;border-radius:15px;background:linear-gradient(120deg,rgba(248,247,255,.96),rgba(240,251,250,.92)); }
        .report-panel [data-testid="stPlotlyChart"] { padding:8px;border:1px solid #ecebf5;border-radius:19px;background:rgba(255,255,255,.74);box-shadow:0 8px 20px rgba(67,73,125,.06); }
        div[data-testid="stExpander"] { border:1px solid #ebe4f4;border-radius:19px;background:rgba(255,255,255,.86);box-shadow:0 8px 20px rgba(72,70,126,.055);transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease;animation:fadeUp .45s ease-out both; }
        div[data-testid="stExpander"]:hover { transform:translateY(-2px);border-color:#d8cef4;box-shadow:0 13px 26px rgba(72,70,126,.1); }
        .flagged-detail { border-radius:16px;padding:16px 18px;margin:4px 0 14px;border:1px solid #f5dcd8;background:linear-gradient(135deg,#fff9f7,#fff1ee); }
        .flagged-detail.low-detail { border-color:#dbe4fb;background:linear-gradient(135deg,#f8faff,#eef3ff); }
        .flagged-detail p { margin:5px 0;color:#505a79;line-height:1.55; }
        .guidance-note { border-left:3px solid #20b9a7;padding:9px 13px;margin:8px 0;background:linear-gradient(100deg,#f0fbf8,#f5f8ff);border-radius:0 12px 12px 0;color:#4f5c78; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#f3efff 0%,#f4f8ff 48%,#effaf8 100%);border-right:1px solid rgba(217,215,239,.8); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#555e7d; }
        .sidebar-brand { margin:8px 0 19px;padding:17px;border:1px solid rgba(255,255,255,.86);border-radius:20px;background:linear-gradient(135deg,rgba(117,87,217,.12),rgba(71,118,230,.1),rgba(32,185,167,.11));box-shadow:0 10px 24px rgba(73,72,137,.08); }
        .sidebar-brand strong { display:block;color:#292b52;font-size:1.15rem;letter-spacing:-.03em; }
        .sidebar-brand span { display:block;margin-top:4px;color:#707897;font-size:.75rem;line-height:1.45; }
        .sidebar-nav-title { margin:0 0 9px;color:#5d55a4;font-size:.68rem;font-weight:850;letter-spacing:.13em;text-transform:uppercase; }
        .sidebar-nav { display:grid;gap:5px;margin:0 0 16px; }
        .sidebar-nav a { display:flex;align-items:center;gap:9px;padding:9px 10px;border:1px solid transparent;border-radius:12px;color:#525c7a;text-decoration:none;font-size:.79rem;font-weight:700;transition:background .18s ease,border-color .18s ease,transform .18s ease; }
        .sidebar-nav a:hover { transform:translateX(3px);border-color:#e5e0f6;background:rgba(255,255,255,.82);color:#654ec0; }
        .sidebar-about { margin-top:16px;padding:13px;border:1px solid rgba(218,218,238,.86);border-radius:15px;background:rgba(255,255,255,.68);color:#707897;font-size:.72rem;line-height:1.5; }
        .sidebar-about strong { display:block;margin-bottom:4px;color:#36385f;font-size:.77rem; }
        .section-heading { scroll-margin-top:1.4rem;display:flex;align-items:center;gap:14px;margin:21px 0 14px;padding:3px 0 13px;border-bottom:1px solid rgba(214,216,237,.82);animation:fadeUp .42s ease-out both; }
        .section-heading-icon { display:grid;place-items:center;flex:0 0 46px;width:46px;height:46px;border:1px solid rgba(255,255,255,.9);border-radius:16px;background:linear-gradient(145deg,#eee7ff,#e5f5ff 58%,#e0fbf4);box-shadow:0 8px 18px rgba(79,86,159,.1);font-size:1.25rem; }
        .section-heading h2 { margin:0;color:#292b52;font-size:clamp(1.25rem,2.4vw,1.65rem);font-weight:820;letter-spacing:-.04em; }
        .section-heading p { margin:3px 0 0;color:#767d98;font-size:.83rem;line-height:1.45; }
        .result-card,.guidance-card,.ingredient-card,.question-card,.download-card { border:1px solid rgba(227,225,244,.92);border-radius:20px;background:linear-gradient(140deg,rgba(255,255,255,.96),rgba(248,249,255,.9));box-shadow:0 11px 26px rgba(71,74,132,.07),inset 0 1px 0 #fff; }
        .result-card { margin:10px 0;padding:19px 21px;border-left:4px solid #ee8070;animation:fadeUp .45s ease-out both; }
        .result-card.low { border-left-color:#6689e8; }
        .result-card-top { display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:13px; }
        .result-card h3,.guidance-card h3,.ingredient-card h3,.question-card h3 { margin:0;color:#2d3058;font-size:1.05rem;font-weight:800;letter-spacing:-.025em; }
        .status-pill { display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border:1px solid #f4d5cf;border-radius:999px;background:#fff2ee;color:#b8514e;font-size:.7rem;font-weight:800;white-space:nowrap; }
        .status-pill.low { border-color:#dbe4fb;background:#f1f5ff;color:#4567bd; }
        .result-meta { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:0 0 14px; }
        .result-meta div { min-width:0;padding:10px 12px;border:1px solid #eeedf6;border-radius:13px;background:rgba(255,255,255,.8); }
        .result-meta small { display:block;color:#858ba4;font-size:.68rem;font-weight:700; }
        .result-meta strong { display:block;margin-top:3px;color:#34375f;font-size:.84rem;overflow-wrap:anywhere; }
        .result-explainer { margin:10px 0 0;padding:11px 13px;border-radius:13px;background:linear-gradient(100deg,#f5f2ff,#f1fbf9);color:#58617d;font-size:.81rem;line-height:1.55; }
        .result-explainer strong { color:#42466d; }
        .guidance-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px; }
        .guidance-item { padding:13px;border:1px solid #ecebf5;border-radius:14px;background:rgba(255,255,255,.79); }
        .guidance-item strong { display:block;margin-bottom:5px;color:#6554b3;font-size:.75rem; }
        .guidance-item p { margin:0;color:#59627f;font-size:.78rem;line-height:1.55; }
        .guidance-card { margin:10px 0;padding:18px 20px;animation:fadeUp .45s ease-out both; }
        .guidance-card h3 { display:flex;align-items:center;gap:8px; }
        .ingredient-notice { margin:10px 0 14px;padding:15px 17px;border:1px solid #f0dfc6;border-radius:16px;background:linear-gradient(110deg,#fffaf0,#fff5f4);color:#675b4e;font-size:.84rem;line-height:1.55; }
        .ingredient-grid { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px; }
        .ingredient-card { padding:17px 18px;animation:fadeUp .45s ease-out both; }
        .ingredient-card h3 { margin-bottom:12px; }
        .ingredient-card p { margin:8px 0;color:#5b6480;font-size:.78rem;line-height:1.55; }
        .ingredient-card p strong { color:#41466c; }
        .ingredient-card a { color:#5f58bb;font-weight:750;text-decoration-thickness:1px;text-underline-offset:2px; }
        .ingredient-card a:hover { color:#238e8a; }
        .question-card { margin:9px 0;padding:15px 18px;display:grid;grid-template-columns:36px 1fr;gap:11px;align-items:start; }
        .question-mark { display:grid;place-items:center;width:34px;height:34px;border-radius:12px;background:linear-gradient(145deg,#eee7ff,#e1f8f3);color:#6756bc;font-weight:850; }
        .question-card p { margin:4px 0 0;color:#59627f;font-size:.82rem;line-height:1.55; }
        .empty-state { padding:17px 19px;border:1px solid #e4e6f4;border-radius:16px;background:linear-gradient(110deg,#f7f5ff,#f0faf8);color:#616987;font-size:.84rem;line-height:1.55; }
        .st-key-complete_results_table { margin:9px 0;padding:7px;border:1px solid #e9e7f3;border-radius:17px;background:rgba(255,255,255,.86); }
        .download-card { margin:8px 0 12px;padding:16px 18px;display:flex;align-items:center;gap:14px; }
        .download-icon { display:grid;place-items:center;flex:0 0 42px;width:42px;height:42px;border-radius:14px;background:linear-gradient(145deg,#7654d4,#4d86e9 65%,#29bba9);color:white;font-size:1.2rem; }
        .download-card strong { display:block;color:#34365d;font-size:.9rem; }
        .download-card span { display:block;margin-top:3px;color:#737b97;font-size:.76rem;line-height:1.45; }
        .footer { margin-top:28px;padding-top:16px;border-top:1px solid rgba(216,218,237,.8);text-align:center;color:#8188a2;font-size:.82rem; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
        @keyframes metricEnter { from{opacity:0;filter:blur(3px)} to{opacity:1;filter:blur(0)} }
        @keyframes floatA { 0%,100%{translate:0 0} 50%{translate:0 -8px} }
        @keyframes floatB { 0%,100%{translate:0 0} 50%{translate:0 7px} }
        @keyframes docHover { 0%,100%{transform:rotateY(-12deg) rotateX(4deg) translateY(0)} 50%{transform:rotateY(-7deg) rotateX(1deg) translateY(-4px)} }
        @keyframes flaskBob { 0%,100%{transform:rotate(-8deg) translateY(0)} 50%{transform:rotate(-4deg) translateY(-4px)} }
        @keyframes bubbleRise { 0%{transform:translateY(9px);opacity:0} 25%{opacity:.9} 100%{transform:translateY(-16px);opacity:0} }
        @keyframes sparkle { 0%,100%{opacity:.45;transform:scale(.85) rotate(-8deg)} 50%{opacity:1;transform:scale(1.12) rotate(8deg)} }
        @media(max-width:900px){ .block-container{padding-left:1rem;padding-right:1rem}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.st-key-upload_workspace [data-testid="stColumn"]{padding:19px 18px 17px}.upload-visual{grid-template-columns:100px 1fr;gap:10px;padding:10px}.demo-scene{grid-template-columns:90px 1fr;padding:8px}.guidance-grid{grid-template-columns:1fr}.ingredient-grid{grid-template-columns:1fr} }
        @media(max-width:620px){ .brandbar{margin-bottom:16px}.st-key-upload_workspace [data-testid="stHorizontalBlock"]{flex-direction:column;gap:14px}.st-key-upload_workspace [data-testid="stColumn"]{width:100%!important;flex:1 1 100%!important}.upload-visual{grid-template-columns:96px 1fr}.summary-grid{gap:10px}.summary-card{min-height:112px;padding:15px;grid-template-columns:43px 1fr;column-gap:10px}.metric-icon{width:42px;height:42px}.summary-card strong{font-size:1.85rem}.result-meta{grid-template-columns:1fr}.section-heading{align-items:flex-start}.section-heading-icon{width:40px;height:40px;flex-basis:40px;border-radius:14px} }
        @media(max-width:390px){ .upload-visual{grid-template-columns:1fr}.doc-scene{height:94px}.workflow{font-size:.61rem}.summary-card{grid-template-columns:1fr;text-align:center;justify-items:center;gap:5px}.metric-icon{grid-row:auto}.summary-card strong{font-size:1.7rem}.summary-card span{margin-top:0} }
        @media(prefers-reduced-motion:reduce){ *,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brandbar" role="banner" aria-label="MediExplain">
      <div class="brand-logo" role="img" aria-label="Overlapping M and E medical logo">
        <span class="brand-logo-m">M</span><span class="brand-logo-e">E</span><span class="brand-logo-pill"></span>
      </div>
      <span class="brand-name">MediExplain</span>
    </div>
    """,
    unsafe_allow_html=True,
)

language = st.selectbox("Display language", ["English", "Hinglish"], label_visibility="collapsed")

st.markdown('<div id="nav-upload-laboratory-report" class="upload-section-anchor"></div>', unsafe_allow_html=True)

with st.container(key="upload_workspace"):
    left, right = st.columns([1.22, 0.78], gap="large")

    with left:
        st.markdown(
            """
            <div class="card upload-card">
              <div class="card-title"><span>📄</span> Upload Laboratory Report</div>
              <div class="upload-visual">
                <div class="doc-scene" aria-hidden="true">
                  <div class="doc-shadow"></div>
                  <div class="doc-sheet"><div class="doc-cross">✚</div><div class="doc-line"></div><div class="doc-line short"></div><div class="doc-line"></div><div class="doc-line short"></div></div>
                  <div class="doc-heart">♥</div><div class="doc-plus">✚</div>
                </div>
                <div class="upload-copy">
                  <strong>Your results, made easier to review</strong>
                  <p>Bring your report in and review detected values before creating a summary.</p>
                  <div class="format-chips"><span class="format-chip">PDF</span><span class="format-chip">JPG</span><span class="format-chip">PNG</span></div>
                  <div class="workflow"><span><b>1</b> Upload</span><i></i><span><b>2</b> Review</span><i></i><span><b>3</b> Understand</span></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        report = st.file_uploader(
            "Choose a PDF, JPG or PNG report",
            type=["pdf", "jpg", "jpeg", "png"],
            help="PDF text will be extracted. JPG and PNG will be previewed; OCR is not enabled.",
        )

    with right:
        st.markdown(
            """
            <div class="card upload-card">
              <div class="card-title"><span>🧪</span> Try Demo Report</div>
              <div class="demo-scene">
                <div class="lab-illustration" aria-hidden="true">
                  <div class="lab-glow"></div><div class="flask"></div>
                  <span class="flask-bubble bubble-one"></span><span class="flask-bubble bubble-two"></span><span class="flask-bubble bubble-three"></span>
                  <span class="lab-spark spark-one">✦</span><span class="lab-spark spark-two">✧</span>
                </div>
                <div class="demo-stats">
                  <div class="demo-stat"><span>Sample tests</span><strong>6</strong></div>
                  <div class="demo-stat"><span>Within range</span><strong class="mint">4</strong></div>
                  <div class="demo-stat"><span>High</span><strong class="coral">2</strong></div>
                </div>
              </div>
              <p class="sample-note">Illustrative sample data only · Not real patient results</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Load 6-test demo report",
            use_container_width=True,
            on_click=load_demo_report,
        )

if report is not None:
    file_extension = report.name.lower().rsplit(".", 1)[-1]
    if file_extension == "pdf":
        report_bytes = report.getvalue()
        report_signature = hashlib.sha256(report_bytes).hexdigest()
        try:
            if report_signature != st.session_state.uploaded_signature:
                with pymupdf.open(stream=report_bytes, filetype="pdf") as doc:
                    extracted_text = "\n".join(page.get_text() for page in doc)
                st.session_state.uploaded_signature = report_signature
                st.session_state.input_signature = report_signature
                st.session_state.report_text = extracted_text
                st.session_state.source_name = report.name
                parsed_rows = parse_report(extracted_text).to_dict("records")
                st.session_state.reviewed_rows = parsed_rows
                st.session_state.parsing_warning = not bool(parsed_rows)
                st.session_state.summary_visible = False
                st.session_state.summary_rows = []
                st.session_state.demo_loaded = False
        except Exception as exc:
            st.error(f"Unable to read this PDF: {type(exc).__name__}")
    else:
        st.markdown('<div class="report-panel">', unsafe_allow_html=True)
        st.success("Image uploaded successfully!")
        st.subheader("Report Preview")
        st.image(report, caption=f"{report.name} · {language}", use_container_width=True)
        st.info("Text extraction from JPG/PNG is not enabled yet.")
        st.markdown("</div>", unsafe_allow_html=True)

extracted_text = st.session_state.report_text
source_name = st.session_state.source_name

if extracted_text.strip():
    st.markdown('<div class="report-panel">', unsafe_allow_html=True)
    render_section_heading(
        "review-extracted-values",
        "🧾",
        "Review Extracted Values",
        "Confirm or correct values, units, and reference ranges before creating a summary.",
    )
    if st.session_state.demo_loaded:
        st.success("Synthetic demo report loaded. Review the extracted text and six detected tests below.")
    else:
        st.success("Report text ready for review")
    st.subheader("Extracted Report Text")
    st.text_area("Report Content", extracted_text, height=240)

    if st.session_state.parsing_warning:
        st.warning(
            "The parser could not detect any supported test results in this report. "
            "No dashboard will be generated until you add or correct a test below."
        )
    if not st.session_state.reviewed_rows:
        st.session_state.reviewed_rows = [
            {"Test Name": "", "Value": None, "Unit": "", "Reference Range": ""}
        ]

    st.subheader("Review & Correct Extracted Values")
    st.caption("Check each value and its report-specific range. Blank or invalid ranges remain 'Cannot Assess'.")

    editor_key = f"test_editor_{st.session_state.input_signature}"
    edited = st.data_editor(
        pd.DataFrame(st.session_state.reviewed_rows),
        key=editor_key,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Test Name": st.column_config.TextColumn(required=True),
            "Value": st.column_config.NumberColumn(format="%.4g"),
            "Unit": st.column_config.TextColumn(),
            "Reference Range": st.column_config.TextColumn(help="Examples: 12 - 16, >5, or <200"),
        },
        hide_index=True,
    )
    st.session_state.reviewed_rows = edited.to_dict("records")

    st.markdown("<br>", unsafe_allow_html=True)
    create_summary = st.button("Create Summary", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if create_summary:
        analyzed = pd.DataFrame(st.session_state.reviewed_rows)
        analyzed["Status"] = [
            classify_value(row["Value"], row["Reference Range"])
            for _, row in analyzed.iterrows()
        ]
        analyzed = analyzed.dropna(subset=["Test Name"])
        analyzed = analyzed[analyzed["Test Name"].astype(str).str.strip() != ""]
        if analyzed.empty:
            st.session_state.summary_rows = []
            st.session_state.summary_visible = False
            st.warning(
                "No test results are available to summarize. Add a test name in the editable table first."
            )
        else:
            st.session_state.summary_rows = analyzed.to_dict("records")
            st.session_state.summary_visible = True

    render_sidebar_navigation(True, st.session_state.summary_visible)

    if st.session_state.summary_visible:
        analyzed = pd.DataFrame(
            st.session_state.summary_rows,
            columns=["Test Name", "Value", "Unit", "Reference Range", "Status"],
        )
        counts = {status: int((analyzed["Status"] == status).sum()) for status in
                  ["Within Range", "High", "Low", "Cannot Assess"]}
        flagged = analyzed[analyzed["Status"].isin(["High", "Low"])]

        render_section_heading(
            "overall-report-summary",
            "📊",
            "Your Summary",
            "A status overview based only on the values and reference ranges you reviewed.",
        )
        if flagged.empty:
            overall_summary = (
                f"Reviewed {len(analyzed)} result(s). No High or Low values were identified using the ranges shown in this report."
                if language == "English"
                else f"{len(analyzed)} result(s) review hue. Report mein diye gaye ranges ke mutabik koi value High ya Low mark nahi hui."
            )
        else:
            flagged_names = ", ".join(
                f"{row['Test Name']} ({row['Status']})" for _, row in flagged.iterrows()
            )
            if language == "English":
                overall_summary = (
                    f"{len(flagged)} result(s) are outside their report-specific ranges: {flagged_names}. "
                    f"{counts['Within Range']} result(s) are Within Range; "
                    f"{counts['Cannot Assess']} could not be assessed. These flags are not a diagnosis."
                )
            else:
                overall_summary = (
                    f"{len(flagged)} result(s) apne report-specific ranges se bahar hain: {flagged_names}. "
                    f"{counts['Within Range']} result(s) Within Range hain; "
                    f"{counts['Cannot Assess']} assess nahi ho sake. Yeh flags diagnosis nahi hain."
                )
        st.info(overall_summary)
        st.caption("These range comparisons are educational and are not a diagnosis.")

        st.markdown(
            f"""
            <div class="summary-grid">
              <div class="summary-card within"><div class="metric-icon">✓</div><strong>{counts["Within Range"]}</strong><span>Within Range</span></div>
              <div class="summary-card high"><div class="metric-icon">↗</div><strong>{counts["High"]}</strong><span>High</span></div>
              <div class="summary-card low"><div class="metric-icon">↘</div><strong>{counts["Low"]}</strong><span>Low</span></div>
              <div class="summary-card unknown"><div class="metric-icon">· · ·</div><strong>{counts["Cannot Assess"]}</strong><span>Cannot Assess</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        status_order = ["Within Range", "High", "Low", "Cannot Assess"]
        status_colors = ["#20b9a7", "#ef6c55", "#4776e6", "#9aa3b2"]
        status_fig = go.Figure(go.Bar(
            x=[counts[status] for status in status_order],
            y=status_order,
            orientation="h",
            marker_color=status_colors,
            text=[counts[status] for status in status_order],
            textposition="outside",
            hovertemplate="%{y}: %{x} result(s)<extra></extra>",
        ))
        status_fig.update_layout(
            title="Result Status Counts",
            xaxis_title="Number of results",
            yaxis_title="",
            template="plotly_white",
            height=330,
            margin=dict(l=20, r=35, t=60, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", color="#59617f"),
            title_font=dict(size=17, color="#292b52"),
            xaxis=dict(dtick=1, rangemode="tozero", gridcolor="#ececf5", zerolinecolor="#d9dced"),
            yaxis=dict(tickfont=dict(color="#535d7a")),
        )
        st.plotly_chart(status_fig, use_container_width=True, config={"displaylogo": False})

        render_section_heading(
            "results-that-need-attention",
            "🔎",
            "Results That Need Attention",
            "Out-of-range results are shown with their own reported value and report-specific range.",
        )
        if flagged.empty:
            st.markdown(
                '<div class="empty-state">No High or Low results were identified using the ranges in this report. A result within range does not replace clinical advice.</div>',
                unsafe_allow_html=True,
            )
        else:
            for _, row in flagged.iterrows():
                test_name = str(row["Test Name"])
                status = str(row["Status"])
                value = "Not provided" if pd.isna(row["Value"]) else str(row["Value"])
                unit = "Not provided" if pd.isna(row["Unit"]) or not str(row["Unit"]).strip() else str(row["Unit"])
                ref_range = "Not provided" if pd.isna(row["Reference Range"]) or not str(row["Reference Range"]).strip() else str(row["Reference Range"])
                guidance = result_guidance(test_name, status, language)
                status_class = status.lower()
                reason = (
                    "The result is above the reference range shown in this report."
                    if status == "High"
                    else "The result is below the reference range shown in this report."
                )
                st.markdown(
                    f"""
                    <article class="result-card {status_class}">
                      <div class="result-card-top">
                        <h3>{html_escape(test_name)}</h3>
                        <span class="status-pill {status_class}">{html_escape(status)}</span>
                      </div>
                      <div class="result-meta">
                        <div><small>Reported value</small><strong>{html_escape(value)} {html_escape(unit)}</strong></div>
                        <div><small>Reference range from report</small><strong>{html_escape(ref_range)}</strong></div>
                        <div><small>Status</small><strong>{html_escape(status)}</strong></div>
                      </div>
                      <p class="result-explainer"><strong>What this test measures:</strong> {html_escape(guidance["measurement"])}</p>
                      <p class="result-explainer"><strong>Why it was flagged:</strong> {reason} This flag alone does not diagnose a condition.</p>
                    </article>
                    """,
                    unsafe_allow_html=True,
                )

        render_section_heading(
            "diet-lifestyle-guidance",
            "🥗",
            "Diet and Lifestyle Guidance",
            "General, test-relevant considerations—not treatment instructions or a substitute for care.",
        )
        if flagged.empty:
            st.markdown(
                '<div class="empty-state">No out-of-range result needs tailored guidance here. Continue routine care and discuss symptoms or concerns with a qualified professional.</div>',
                unsafe_allow_html=True,
            )
        else:
            for _, row in flagged.iterrows():
                test_name = str(row["Test Name"])
                guidance = result_guidance(test_name, str(row["Status"]), language)
                st.markdown(
                    f"""
                    <article class="guidance-card">
                      <h3>✦ {html_escape(test_name)}</h3>
                      <div class="guidance-grid">
                        <div class="guidance-item"><strong>Food</strong><p>{html_escape(guidance["diet"])}</p></div>
                        <div class="guidance-item"><strong>Movement</strong><p>{html_escape(guidance["activity"])}</p></div>
                        <div class="guidance-item"><strong>Lifestyle and follow-up</strong><p>{html_escape(guidance["lifestyle"])}</p></div>
                      </div>
                    </article>
                    """,
                    unsafe_allow_html=True,
                )

        render_section_heading(
            "medication-ingredient-information",
            "💊",
            "Medication Ingredient Information",
            "General ingredient education for clinician-confirmed deficiencies; this section is not matched to your results.",
        )
        st.markdown(
            """
            <div class="ingredient-notice">
              <strong>No medication can be recommended from this report alone. Discuss these results with a qualified clinician.</strong>
              <br>These examples apply only after a clinician confirms the relevant deficiency. A flagged hemoglobin or vitamin result does not establish a deficiency. No ingredients are automatically suggested for kidney, glucose, or other findings requiring individual assessment.
            </div>
            """,
            unsafe_allow_html=True,
        )
        ingredient_cards = []
        for item in MEDICATION_INGREDIENT_INFO:
            ingredient_cards.append(
                '<article class="ingredient-card">'
                f'<h3>{html_escape(item["ingredient"])}</h3>'
                f'<p><strong>General purpose:</strong> {html_escape(item["purpose"])}</p>'
                f'<p><strong>Used for:</strong> {html_escape(item["condition"])}</p>'
                f'<p><strong>Why evaluation matters:</strong> {html_escape(item["evaluation"])}</p>'
                f'<p><strong>Important precautions:</strong> {html_escape(item["precautions"])}</p>'
                f'<p><a href="{html_escape(item["reference"])}" target="_blank" rel="noopener noreferrer">{html_escape(item["reference_label"])}</a></p>'
                '</article>'
            )
        st.markdown(
            '<div class="ingredient-grid">\n' + "\n".join(ingredient_cards) + "\n</div>",
            unsafe_allow_html=True,
        )

        render_section_heading(
            "questions-for-your-doctor",
            "❔",
            "Questions for Your Doctor",
            "Use these prompts to discuss report-specific results and the clinical context.",
        )
        if flagged.empty:
            st.markdown(
                '<div class="empty-state">What do these results mean in the context of my symptoms, medical history, and other tests?</div>',
                unsafe_allow_html=True,
            )
        else:
            for _, row in flagged.iterrows():
                test_name = str(row["Test Name"])
                guidance = result_guidance(test_name, str(row["Status"]), language)
                st.markdown(
                    f"""
                    <article class="question-card">
                      <div class="question-mark">?</div>
                      <div><h3>{html_escape(test_name)}</h3><p>{html_escape(guidance["doctor_question"])}</p></div>
                    </article>
                    """,
                    unsafe_allow_html=True,
                )
        st.caption("A qualified healthcare professional can interpret these results with your symptoms and history.")

        render_section_heading(
            "complete-test-results",
            "📋",
            "Complete Test Results",
            "All reviewed test names, reported values, units, reference ranges, and statuses.",
        )
        with st.container(key="complete_results_table"):
            st.dataframe(analyzed, hide_index=True, use_container_width=True)

        render_section_heading(
            "download-summary",
            "⬇️",
            "Download Summary",
            "Save a text copy of the reviewed results and status counts.",
        )
        st.markdown(
            """
            <div class="download-card">
              <div class="download-icon">↓</div>
              <div><strong>Your report summary</strong><span>Includes reviewed values, report-provided ranges, status counts, and the educational disclaimer.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_lines = [
            "MediExplain - Concise Report Summary",
            f"Source: {st.session_state.source_name or 'Uploaded report'}",
            f"Language: {language}",
            "",
            f"Within Range: {counts['Within Range']}",
            f"High: {counts['High']}",
            f"Low: {counts['Low']}",
            f"Cannot Assess: {counts['Cannot Assess']}",
            "",
            "Reviewed values:",
        ]
        for _, row in analyzed.iterrows():
            value = "Not provided" if pd.isna(row["Value"]) else str(row["Value"])
            unit = "" if pd.isna(row["Unit"]) else str(row["Unit"])
            ref_range = "Not provided" if pd.isna(row["Reference Range"]) or not str(row["Reference Range"]).strip() else str(row["Reference Range"])
            summary_lines.append(
                f"- {row['Test Name']}: {value} {unit} "
                f"| Range: {ref_range} | {row['Status']}"
            )
        summary_lines.extend([
            "",
            "Educational information only. Not a diagnosis or prescription and not a substitute for professional medical advice.",
            "No medication can be recommended from this report alone. Discuss these results with a qualified clinician.",
        ])

        st.download_button(
            "Download TXT Summary",
            data="\n".join(summary_lines),
            file_name="mediexplain_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
else:
    render_sidebar_navigation(False, False)

with st.expander("NLP & ML Demonstration", expanded=False):
    st.caption(
        "Synthetic educational statements only. This text classifier predicts a topic category; "
        "it does not classify patient health, diagnose conditions, or recommend treatment."
    )
    evaluation = get_demo_evaluation()
    metric_columns = st.columns(4)
    metric_values = (
        ("Accuracy", evaluation.accuracy),
        ("Precision (macro)", evaluation.precision),
        ("Recall (macro)", evaluation.recall),
        ("F1-score (macro)", evaluation.f1_score),
    )
    for column, (label, score) in zip(metric_columns, metric_values):
        column.metric(label, f"{score:.1%}")

    st.caption(
        f"Demonstration-only evaluation on synthetic data: "
        f"{evaluation.training_examples} training statements and "
        f"{evaluation.test_examples} stratified holdout statements "
        f"({len(SYNTHETIC_EXAMPLES)} total)."
    )
    st.subheader("Confusion Matrix")
    confusion_frame = pd.DataFrame(
        evaluation.confusion_matrix,
        index=evaluation.category_labels,
        columns=evaluation.category_labels,
    )
    confusion_frame.index.name = "Actual category"
    st.dataframe(confusion_frame, use_container_width=True)

    statement = st.text_input(
        "Try an educational health statement",
        key="ml_demo_statement",
        help="Enter an educational statement topic, not a patient result or request for diagnosis.",
    )
    if statement.strip():
        predicted_category = predict_category(statement, evaluation.model)
        st.info(f"Predicted educational category: {predicted_category}")

st.markdown(
    '<div class="footer">MediExplain · Educational information only · Not a diagnostic or prescription system</div>',
    unsafe_allow_html=True,
)
