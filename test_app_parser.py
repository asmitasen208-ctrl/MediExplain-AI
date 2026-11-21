import unittest
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from app import (
    _REPORT_WARNING_BACKGROUND,
    _REPORT_WARNING_BORDER,
    _REPORT_WARNING_TEXT,
    _normalize_ocr_per_ul_unit,
    _ocr_image_with_raw,
    extract_image_text,
    _validate_report_rows,
    extract_lab_information,
)
from modules.report_interpreter import interpret_report_rows


class ReportParserRegressionTests(unittest.TestCase):
    def test_six_row_ocr_sample_keeps_fields_with_their_rows(self):
        report_text = (
            "Haemoglobin 14.2 g/dL 13.0 - 17.0\n"
            "WBC 7,200 /uL 4,000 - 11,000\n"
            "Platelets 250,000 /uL 150,000 - 450,000\n"
            "RBC 4.8 million/uL 4.5 - 5.9\n"
            "Blood Glucose (Fasting) 96 mg/dL 70 - 99\n"
            "HbA1c 5.4 % Below 5.7"
        )

        rows = extract_lab_information(report_text)

        self.assertEqual(
            [(row["Test name"], row["Result value"], row["Unit"], row["Reference range"]) for row in rows],
            [
                ("Haemoglobin", "14.2", "g/dL", "13.0 - 17.0"),
                ("WBC", "7,200", "/uL", "4,000 - 11,000"),
                ("Platelets", "250,000", "/uL", "150,000 - 450,000"),
                ("RBC", "4.8", "million/uL", "4.5 - 5.9"),
                ("Blood Glucose (Fasting)", "96", "mg/dL", "70 - 99"),
                ("HbA1c", "5.4", "%", "Below 5.7"),
            ],
        )

    def test_hba1c_does_not_receive_haemoglobin_range(self):
        report_text = (
            "Haemoglobin: 14.2 g/dL\n"
            "Reference Range: 13.0 - 17.0\n"
            "HbA1c: 5.4 %"
        )

        rows = extract_lab_information(report_text)
        hba1c = next(row for row in rows if row["Test name"] == "HbA1c")

        self.assertEqual(hba1c["Result value"], "5.4")
        self.assertEqual(hba1c["Unit"], "%")
        self.assertEqual(hba1c["Reference range"], "Unavailable")

    def test_missing_inline_fields_remain_unavailable(self):
        report_text = "WBC 7,200\nRBC 4.8 million/uL"

        rows = extract_lab_information(report_text)

        wbc = next(row for row in rows if row["Test name"] == "WBC")
        rbc = next(row for row in rows if row["Test name"] == "RBC")
        self.assertEqual(wbc["Result value"], "7,200")
        self.assertEqual(wbc["Unit"], "Unavailable")
        self.assertEqual(wbc["Reference range"], "Unavailable")
        self.assertEqual(rbc["Reference range"], "Unavailable")

    def test_ocr_style_table_columns_stay_in_their_own_columns(self):
        report_text = (
            "Test Name                    Result       Unit          Reference Range\n"
            "Haemoglobin                  14.2         g/dL          13.0 - 17.0\n"
            "WBC                          7,200                      4,000 - 11,000\n"
            "Platelets                    250,000      /uL           150,000 - 450,000\n"
            "RBC                          4.8          million/uL    4.5 - 5.9\n"
            "Blood Glucose (Fasting)      96           mg/dL         70 - 99\n"
            "HbA1c                        5.4          %             Below 5.7"
        )

        rows = extract_lab_information(report_text)

        wbc = next(row for row in rows if row["Test name"] == "WBC")
        hba1c = next(row for row in rows if row["Test name"] == "HbA1c")

        self.assertEqual(wbc["Result value"], "7,200")
        self.assertEqual(wbc["Unit"], "Unavailable")
        self.assertEqual(wbc["Reference range"], "4,000 - 11,000")
        self.assertEqual(hba1c["Reference range"], "Below 5.7")
        self.assertNotEqual(hba1c["Reference range"], "13.0 - 17.0")

        issues = dict(_validate_report_rows(rows))
        self.assertIn("Unit is missing.", issues[2])
        self.assertNotIn("Unit is missing.", issues.get(1, []))
        self.assertNotIn("Unit is missing.", issues.get(6, []))

    def test_missing_range_is_unavailable_and_flagged_for_manual_review(self):
        rows = [
            {
                "Test name": "WBC",
                "Result value": "7,200",
                "Unit": "/uL",
                "Reference range": "Unavailable",
            }
        ]

        issues = dict(_validate_report_rows(rows))

        self.assertEqual(rows[0]["Reference range"], "Unavailable")
        self.assertIn("Reference range is missing.", issues[1])

    def test_exact_six_test_ocr_output_maps_ranges_positionally(self):
        report_text = (
            "Haemoglobin 14.2 g/dL\n"
            "WBC 7,200 Jul\n"
            "Platelets 250,000 /uL\n"
            "RBC 4.8 million/uL\n"
            "Blood Glucose (Fasting) 96 mg/dL\n"
            "HbA1c 5.4 %\n"
            "\n"
            "REFERENCE RANGE\n"
            "\n"
            "13.0 - 17.0\n"
            "4,000 - 11,000\n"
            "150,000 - 450,000\n"
            "4.5-5.9\n"
            "70 - 99\n"
            "Below 5.7"
        )

        rows = extract_lab_information(report_text)
        ranges = [row["Reference range"] for row in rows]

        self.assertEqual(
            ranges,
            [
                "13.0 - 17.0",
                "4,000 - 11,000",
                "150,000 - 450,000",
                "4.5-5.9",
                "70 - 99",
                "Below 5.7",
            ],
        )
        wbc = next(row for row in rows if row["Test name"] == "WBC")
        hba1c = next(row for row in rows if row["Test name"] == "HbA1c")
        self.assertEqual(wbc["Unit"], "/uL")
        self.assertEqual(hba1c["Reference range"], "Below 5.7")
        self.assertNotEqual(hba1c["Reference range"], "13.0 - 17.0")

        issues = dict(_validate_report_rows(rows))
        self.assertNotIn(2, issues)

    def test_five_ranges_for_six_tests_are_not_guessed(self):
        report_text = (
            "Haemoglobin 14.2 g/dL\n"
            "WBC 7,200 /uL\n"
            "Platelets 250,000 /uL\n"
            "RBC 4.8 million/uL\n"
            "Blood Glucose (Fasting) 96 mg/dL\n"
            "HbA1c 5.4 %\n"
            "REFERENCE RANGE\n"
            "13.0 - 17.0\n"
            "4,000 - 11,000\n"
            "150,000 - 450,000\n"
            "4.5-5.9\n"
            "70 - 99"
        )

        rows = extract_lab_information(report_text)

        self.assertTrue(rows)
        self.assertTrue(
            all(row["Reference range"] == "Unavailable" for row in rows)
        )
        issues = dict(_validate_report_rows(rows))
        self.assertTrue(all("Reference range is missing." in issues[i] for i in range(1, 7)))

    def test_extra_range_entries_are_not_mapped(self):
        report_text = (
            "Haemoglobin 14.2 g/dL\n"
            "WBC 7,200 /uL\n"
            "Platelets 250,000 /uL\n"
            "RBC 4.8 million/uL\n"
            "Blood Glucose (Fasting) 96 mg/dL\n"
            "HbA1c 5.4 %\n"
            "REFERENCE RANGE\n"
            "13.0 - 17.0\n"
            "4,000 - 11,000\n"
            "150,000 - 450,000\n"
            "4.5-5.9\n"
            "70 - 99\n"
            "Below 5.7\n"
            "0 - 1"
        )

        rows = extract_lab_information(report_text)

        self.assertTrue(
            all(row["Reference range"] == "Unavailable" for row in rows)
        )

    def test_inline_ranges_still_work_without_positional_override(self):
        report_text = (
            "Haemoglobin: 14.2 g/dL\n"
            "Reference Range: 13.0 - 17.0\n"
            "WBC: 7,200 /uL\n"
            "Reference Range: 4,000 - 11,000"
        )

        rows = extract_lab_information(report_text)

        haemoglobin = next(row for row in rows if row["Test name"] == "Haemoglobin")
        wbc = next(row for row in rows if row["Test name"] == "WBC")
        self.assertEqual(haemoglobin["Reference range"], "13.0 - 17.0")
        self.assertEqual(wbc["Reference range"], "4,000 - 11,000")

    def test_ambiguous_range_section_remains_unavailable_and_flagged(self):
        report_text = (
            "Haemoglobin 14.2 g/dL\n"
            "WBC 7,200 /uL\n"
            "REFERENCE RANGE\n"
            "13.0 - 17.0\n"
            "Remark 42"
        )

        rows = extract_lab_information(report_text)

        self.assertTrue(
            all(row["Reference range"] == "Unavailable" for row in rows)
        )
        issues = dict(_validate_report_rows(rows))
        self.assertTrue(
            all("Reference range is missing." in issues[i] for i in (1, 2))
        )

    def test_wbc_per_ul_is_compared_in_report_units(self):
        row = {
            "Test name": "WBC",
            "Result value": "7,200",
            "Unit": "/uL",
            "Reference range": "4,000 - 11,000 /uL",
        }

        interpretation = interpret_report_rows([row])[0]

        self.assertEqual(interpretation["Classification"], "Within reference range")

    def test_platelets_per_ul_is_compared_in_report_units(self):
        row = {
            "Test name": "Platelets",
            "Result value": "250,000",
            "Unit": "/uL",
            "Reference range": "150,000 - 450,000 /uL",
        }

        interpretation = interpret_report_rows([row])[0]

        self.assertEqual(interpretation["Classification"], "Within reference range")

    def test_unsupported_unit_remains_unclassified(self):
        row = {
            "Test name": "WBC",
            "Result value": "7,200",
            "Unit": "mg/dL",
            "Reference range": "4,000 - 11,000 /uL",
        }

        interpretation = interpret_report_rows([row])[0]

        self.assertEqual(interpretation["Classification"], "Unable to classify")

    def test_incompatible_scales_are_not_compared(self):
        row = {
            "Test name": "WBC",
            "Result value": "7.2",
            "Unit": "10^3/uL",
            "Reference range": "4,000 - 11,000 /uL",
        }

        interpretation = interpret_report_rows([row])[0]

        self.assertEqual(interpretation["Classification"], "Unable to classify")
        self.assertIn("incompatible", interpretation["Notes"].lower())

    def test_hba1c_clinical_threshold_remains_unclassified(self):
        row = {
            "Test name": "HbA1c",
            "Result value": "5.4",
            "Unit": "%",
            "Reference range": "Below 5.7",
        }

        interpretation = interpret_report_rows([row])[0]

        self.assertEqual(interpretation["Classification"], "Unable to classify")
        self.assertIn("clinical_decision_threshold", interpretation["Notes"])

    def test_haemoglobin_and_rbc_behavior_is_unchanged(self):
        rows = [
            {
                "Test name": "Haemoglobin",
                "Result value": "14.2",
                "Unit": "g/dL",
                "Reference range": "13.0 - 17.0",
            },
            {
                "Test name": "RBC",
                "Result value": "4.8",
                "Unit": "million/uL",
                "Reference range": "4.5 - 5.9",
            },
        ]

        interpretations = interpret_report_rows(rows)

        self.assertEqual(
            [row["Classification"] for row in interpretations],
            ["Within reference range", "Within reference range"],
        )

    def test_wbc_and_platelet_ocr_unit_normalization_is_contextual(self):
        self.assertEqual(
            _normalize_ocr_per_ul_unit("WBC", "Jul"),
            "/uL",
        )
        self.assertEqual(
            _normalize_ocr_per_ul_unit("Platelets", "/ul"),
            "/uL",
        )
        self.assertEqual(
            _normalize_ocr_per_ul_unit("Haemoglobin", "Jul"),
            "Jul",
        )
        self.assertEqual(
            _normalize_ocr_per_ul_unit("WBC", "some unclear unit"),
            "some unclear unit",
        )

    def test_wbc_jul_row_has_no_suspicious_unit_issue(self):
        report_text = "WBC 7,200 Jul\n"

        rows = extract_lab_information(report_text)

        wbc = next(row for row in rows if row["Test name"] == "WBC")
        issues = dict(_validate_report_rows(rows))
        self.assertEqual(wbc["Unit"], "/uL")
        row_issues = issues.get(1, [])
        self.assertFalse(
            any("Unit" in issue or "suspicious" in issue for issue in row_issues)
        )

    def test_report_warning_colors_are_readable_and_consistent(self):
        self.assertEqual(_REPORT_WARNING_BACKGROUND, "#FFF4CC")
        self.assertEqual(_REPORT_WARNING_TEXT, "#333333")
        self.assertEqual(_REPORT_WARNING_BORDER, "#E0B84C")

    def test_in_memory_sample_image_uses_complete_ocr_pipeline(self):
        image = Image.new("RGB", (1600, 900), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 34)
        except OSError:
            font = ImageFont.load_default()

        lines = [
            "Haemoglobin 14.2 g/dL",
            "WBC 7,200 /uL",
            "Platelets 250,000 /uL",
            "RBC 4.8 million/uL",
            "Blood Glucose (Fasting) 96 mg/dL",
            "HbA1c 5.4 %",
            "REFERENCE RANGE",
            "13.0 - 17.0",
            "4,000 - 11,000",
            "150,000 - 450,000",
            "4.5-5.9",
            "70 - 99",
            "Below 5.7",
        ]
        y = 30
        for line in lines:
            draw.text((50, y), line, font=font, fill="black")
            y += 58

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        improved_text = extract_image_text(image_bytes.getvalue())
        raw_text = getattr(
            __import__("streamlit").session_state,
            "raw_ocr_text_debug",
            improved_text,
        )
        rows = extract_lab_information(improved_text)
        row_by_name = {row["Test name"]: row for row in rows}

        self.assertIn("WBC", row_by_name)
        self.assertEqual(row_by_name["WBC"]["Unit"], "/uL")
        self.assertEqual(row_by_name["WBC"]["Result value"], "7,200")
        self.assertEqual(row_by_name["Platelets"]["Unit"], "/uL")
        self.assertEqual(row_by_name["Platelets"]["Result value"], "250,000")
        self.assertEqual(row_by_name["Haemoglobin"]["Result value"], "14.2")
        self.assertEqual(row_by_name["RBC"]["Result value"], "4.8")
        self.assertEqual(row_by_name["Blood Glucose (Fasting)"]["Result value"], "96")
        self.assertEqual(row_by_name["HbA1c"]["Result value"], "5.4")
        self.assertTrue(raw_text)


if __name__ == "__main__":
    unittest.main()
