"""Synthetic, educational text-classification demonstration for MediExplain."""

from dataclasses import dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as calculate_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


CATEGORY_LABELS = (
    "Nutrition",
    "Exercise",
    "Laboratory Reference Ranges",
    "General Wellness",
)

SYNTHETIC_EXAMPLES = (
    ("Nutrition", "A balanced meal can include vegetables, fruit, whole grains, and legumes."),
    ("Nutrition", "Nutrition education encourages varied foods across the main food groups."),
    ("Nutrition", "Beans and lentils are examples of fiber-containing foods."),
    ("Nutrition", "Food labels can help compare fiber and sodium in packaged foods."),
    ("Nutrition", "Meal planning can include different vegetables and fruit throughout the week."),
    ("Nutrition", "Whole grains are one option when choosing foods with dietary fiber."),
    ("Nutrition", "Nutrition guidance can discuss serving sizes and food variety."),
    ("Nutrition", "Cooking at home may make it easier to include vegetables in meals."),
    ("Nutrition", "A varied eating pattern includes more than one type of food."),
    ("Nutrition", "Reading ingredient lists is one part of learning about food choices."),
    ("Nutrition", "Legumes, vegetables, and grains can be combined in balanced meals."),
    ("Nutrition", "Nutrition topics include meal choices, food groups, and dietary variety."),
    ("Exercise", "Regular walking is one example of moderate physical activity."),
    ("Exercise", "Movement breaks can interrupt long periods of sitting."),
    ("Exercise", "Exercise routines may include aerobic, strength, and balance activities."),
    ("Exercise", "Choose activity that suits your abilities and increase it gradually."),
    ("Exercise", "Stretching and mobility work can be part of a movement routine."),
    ("Exercise", "Cycling, dancing, and swimming are examples of physical activity."),
    ("Exercise", "An activity plan can include both movement and recovery days."),
    ("Exercise", "A comfortable pace is useful when beginning a walking routine."),
    ("Exercise", "Warm-ups and cool-downs are common parts of exercise sessions."),
    ("Exercise", "Strength exercise involves using muscles against resistance."),
    ("Exercise", "Short activity breaks can add movement to a sedentary day."),
    ("Exercise", "Physical activity goals should be adapted to mobility and comfort."),
    ("Laboratory Reference Ranges", "A laboratory reference range is the interval printed for a test."),
    ("Laboratory Reference Ranges", "Compare a lab value with the reference interval on that report."),
    ("Laboratory Reference Ranges", "Reference intervals can vary between laboratories and methods."),
    ("Laboratory Reference Ranges", "A result outside its reference range is not a diagnosis by itself."),
    ("Laboratory Reference Ranges", "Review the units and range shown beside the laboratory test."),
    ("Laboratory Reference Ranges", "A report may mark a result high or low against its stated range."),
    ("Laboratory Reference Ranges", "Use the report-specific range when reviewing a laboratory value."),
    ("Laboratory Reference Ranges", "Check that the value and units match the correct lab test."),
    ("Laboratory Reference Ranges", "Reference ranges may depend on the laboratory and testing context."),
    ("Laboratory Reference Ranges", "A result within range does not replace professional advice."),
    ("Laboratory Reference Ranges", "Laboratory results need context from other findings and history."),
    ("Laboratory Reference Ranges", "Test reports provide reference intervals for individual measurements."),
    ("General Wellness", "Consistent sleep routines are one part of general wellbeing."),
    ("General Wellness", "Stress management can include relaxation and social support."),
    ("General Wellness", "General wellness includes rest, hydration, and everyday routines."),
    ("General Wellness", "Taking regular breaks can support comfort during desk work."),
    ("General Wellness", "Social connection and asking for support can benefit wellbeing."),
    ("General Wellness", "A regular sleep schedule is a general wellness habit."),
    ("General Wellness", "Preventive care is one topic in general health education."),
    ("General Wellness", "Small sustainable habits can support overall wellbeing."),
    ("General Wellness", "Make time for enjoyable activities and adequate rest."),
    ("General Wellness", "Wellness guidance should fit a person's circumstances."),
    ("General Wellness", "Daily routines can include rest, recreation, and time outdoors."),
    ("General Wellness", "General wellness information is educational, not personal medical advice."),
)


@dataclass(frozen=True)
class DemoEvaluation:
    """Trained text model and its held-out synthetic demonstration metrics."""

    model: Pipeline
    category_labels: tuple[str, ...]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    training_examples: int
    test_examples: int


def train_and_evaluate() -> DemoEvaluation:
    """Train a TF-IDF/logistic-regression pipeline and evaluate a stratified holdout."""
    statements = [statement for _, statement in SYNTHETIC_EXAMPLES]
    categories = [category for category, _ in SYNTHETIC_EXAMPLES]
    train_statements, test_statements, train_categories, test_categories = train_test_split(
        statements,
        categories,
        test_size=0.25,
        random_state=42,
        stratify=categories,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")),
            ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(train_statements, train_categories)
    predictions = model.predict(test_statements)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_categories,
        predictions,
        labels=CATEGORY_LABELS,
        average="macro",
        zero_division=0,
    )
    matrix = calculate_confusion_matrix(
        test_categories,
        predictions,
        labels=CATEGORY_LABELS,
    )

    return DemoEvaluation(
        model=model,
        category_labels=CATEGORY_LABELS,
        accuracy=float(accuracy_score(test_categories, predictions)),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in matrix.tolist()
        ),
        training_examples=len(train_statements),
        test_examples=len(test_statements),
    )


@lru_cache(maxsize=1)
def get_demo_evaluation() -> DemoEvaluation:
    """Return one cached training/evaluation run for Streamlit reruns."""
    return train_and_evaluate()


def predict_category(statement: str, model: Pipeline | None = None) -> str:
    """Predict an educational statement topic, never a patient's health condition."""
    cleaned_statement = statement.strip()
    if not cleaned_statement:
        raise ValueError("Enter a non-empty educational statement.")

    active_model = model if model is not None else get_demo_evaluation().model
    return str(active_model.predict([cleaned_statement])[0])
