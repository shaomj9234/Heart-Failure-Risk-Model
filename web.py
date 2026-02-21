import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import shap
except Exception:
    shap = None

st.set_page_config(
    page_title="Circadian Syndrome-Heart Failure Risk Model interfaceV1.0",
    page_icon="HF",
    layout="wide",
)

APP_TITLE = "Circadian Syndrome-Heart Failure Risk Model interfaceV1.0"
APP_SUBTITLE = "Survival risk prediction based on GradientBoostingSurvivalAnalysis."

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "CSGBDTSmodel.dat"

FEATURE_SPECS = [
    {
        "model_key": "Circadian_syndrome",
        "label": "Circadian syndrome",
        "desc": "Yes/No",
        "input_type": "binary_yes_no",
    },
    {
        "model_key": "Verdugo",
        "label": "Vericiguat use",
        "desc": "Yes/No",
        "input_type": "binary_yes_no",
    },
    {
        "model_key": "Ischemic_cardiomyopathy",
        "label": "Ischemic cardiomyopathy",
        "desc": "Yes/No",
        "input_type": "binary_yes_no",
    },
    {
        "model_key": "HbA1c",
        "label": "HbA1c (%)",
        "desc": "Range in document: 4.0~100",
        "input_type": "number",
        "min": 4.0,
        "max": 100.0,
        "step": 0.1,
    },
    {
        "model_key": "eGFR",
        "label": "eGFR (mL/min/1.73 m2)",
        "desc": "Range in document: 0~150",
        "input_type": "number",
        "min": 0.0,
        "max": 150.0,
        "step": 0.1,
    },
    {
        "model_key": "HR",
        "label": "Heart Rate (bpm)",
        "desc": "Range in document: 20~250",
        "input_type": "number",
        "min": 20.0,
        "max": 250.0,
        "step": 1.0,
    },
    {
        "model_key": "Age",
        "label": "Age (years)",
        "desc": "Range in document: 18~120",
        "input_type": "number",
        "min": 18.0,
        "max": 120.0,
        "step": 1.0,
    },
    {
        "model_key": "SII",
        "label": "Systemic immune-inflammation index (SII)",
        "desc": "Range in document: 1~5000",
        "input_type": "number",
        "min": 1.0,
        "max": 5000.0,
        "step": 1.0,
    },
    {
        "model_key": "NTproBNP",
        "label": "NT-proBNP (pg/mL)",
        "desc": "Range in document: 100~35000",
        "input_type": "number",
        "min": 100.0,
        "max": 35000.0,
        "step": 10.0,
    },
    {
        "model_key": "LVEF",
        "label": "LVEF (%)",
        "desc": "Range in document: 10~80",
        "input_type": "number",
        "min": 10.0,
        "max": 80.0,
        "step": 0.1,
    },
    {
        "model_key": "TyG",
        "label": "Triglycerides index (TyG)",
        "desc": "Range in document: 0.1~200",
        "input_type": "number",
        "min": 0.1,
        "max": 200.0,
        "step": 0.01,
    },
    {
        "model_key": "Female",
        "label": "Gender",
        "desc": "Female/Male (encoded as Female=1, Male=0)",
        "input_type": "female_male",
    },
    {
        "model_key": "BB",
        "label": "beta-Blocker use",
        "desc": "Yes/No",
        "input_type": "binary_yes_no",
    },
]


@st.cache_resource
def load_model() -> Any:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    with MODEL_FILE.open("rb") as f:
        return pickle.load(f)


def render_sidebar() -> None:
    st.sidebar.title("System Info")
    st.sidebar.markdown(f"**System:** {APP_TITLE}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Included Variables")
    for spec in FEATURE_SPECS:
        st.sidebar.markdown(f"- **{spec['label']}**: {spec['desc']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Notes")
    st.sidebar.markdown(
        "- TyG = Ln [triglycerides (mg/dL) x glucose (mg/dL) / 2]\n"
        "- SII = (Neutrophil Count x Platelet Count) / Lymphocyte Count"
    )


def _default_numeric(spec: dict[str, Any]) -> float:
    return float((spec["min"] + spec["max"]) / 2.0)


def build_input_form() -> tuple[bool, dict[str, float]]:
    user_inputs: dict[str, float] = {}
    with st.form("predict_form"):
        cols = st.columns(2)
        for i, spec in enumerate(FEATURE_SPECS):
            key = spec["model_key"]
            with cols[i % 2]:
                if spec["input_type"] == "number":
                    user_inputs[key] = st.number_input(
                        label=spec["label"],
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=_default_numeric(spec),
                        step=float(spec["step"]),
                        format="%.4f",
                        help=spec["desc"],
                        key=f"input_{key}",
                    )
                elif spec["input_type"] == "binary_yes_no":
                    option = st.selectbox(
                        spec["label"],
                        options=["No", "Yes"],
                        index=0,
                        help=spec["desc"],
                        key=f"input_{key}",
                    )
                    user_inputs[key] = 1.0 if option == "Yes" else 0.0
                elif spec["input_type"] == "female_male":
                    option = st.selectbox(
                        spec["label"],
                        options=["Male", "Female"],
                        index=0,
                        help=spec["desc"],
                        key=f"input_{key}",
                    )
                    user_inputs[key] = 1.0 if option == "Female" else 0.0

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)
    return submitted, user_inputs


def _predict_risk_values(model: Any, model_features: list[str], data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        data_df = data.copy()
    else:
        data_df = pd.DataFrame(data, columns=model_features)
    return np.asarray(model.predict(data_df), dtype=float)


def build_shap_background(input_df: pd.DataFrame, model_features: list[str], n_samples: int = 80) -> pd.DataFrame:
    spec_by_key = {s["model_key"]: s for s in FEATURE_SPECS}
    base_row = input_df.iloc[0]
    rng = np.random.default_rng(42)

    rows: list[dict[str, float]] = []
    for _ in range(n_samples):
        row: dict[str, float] = {}
        for key in model_features:
            base_val = float(base_row[key])
            spec = spec_by_key.get(key)
            if spec is None:
                row[key] = base_val
                continue

            input_type = spec.get("input_type")
            if input_type == "number":
                lo = float(spec["min"])
                hi = float(spec["max"])
                span = max(hi - lo, 1e-6)
                sigma = max(0.08 * span, float(spec.get("step", 1.0)))
                val = float(rng.normal(base_val, sigma))
                row[key] = float(np.clip(val, lo, hi))
            elif input_type in {"binary_yes_no", "female_male"}:
                flip = rng.random() < 0.2
                row[key] = 1.0 - base_val if flip else base_val
            else:
                row[key] = base_val
        rows.append(row)

    bg_df = pd.DataFrame(rows, columns=model_features)
    bg_df.iloc[0] = input_df.iloc[0]
    return bg_df


def render_shap_force_plot(model: Any, model_features: list[str], input_df: pd.DataFrame) -> None:
    st.subheader("SHAP Force Plot")
    if shap is None:
        st.warning("`shap` is not installed in the current environment.")
        return

    bg_df = build_shap_background(input_df, model_features)

    predict_fn = lambda x: _predict_risk_values(model, model_features, x)
    try:
        with st.spinner("Computing SHAP force plot..."):
            explainer = shap.KernelExplainer(predict_fn, bg_df)
            try:
                shap_values = explainer.shap_values(input_df, nsamples=120, silent=True)
            except TypeError:
                shap_values = explainer.shap_values(input_df, nsamples=120)

        if isinstance(shap_values, list):
            shap_row = np.asarray(shap_values[0])[0]
        else:
            shap_array = np.asarray(shap_values)
            shap_row = shap_array if shap_array.ndim == 1 else shap_array[0]

        if float(np.sum(np.abs(shap_row))) < 1e-9:
            st.warning(
                "SHAP values are near zero for this input/background setup. "
                "Try adjusting inputs to get a more informative force plot."
            )

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.asarray(expected_value).reshape(-1)[0])
        else:
            expected_value = float(expected_value)

        try:
            force_plot = shap.force_plot(
                expected_value,
                shap_row,
                input_df.iloc[0],
                feature_names=model_features,
                matplotlib=False,
                show=False,
            )
        except TypeError:
            force_plot = shap.force_plot(
                expected_value,
                shap_row,
                input_df.iloc[0],
                feature_names=model_features,
                matplotlib=False,
            )

        if hasattr(force_plot, "html"):
            html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html, height=420, scrolling=False)
        else:
            st.warning("Failed to render SHAP force plot in HTML mode.")
    except Exception as exc:
        st.warning(f"SHAP force plot generation failed: {exc}")


def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.info(
            "This model requires scikit-survival + compatible scikit-learn.\n"
            "Please install packages in requirements.txt first."
        )
        return

    render_sidebar()

    st.subheader("Patient Input")
    st.write("Please enter all model variables. Inputs are constrained by the ranges from your document.")
    submitted, user_inputs = build_input_form()
    if not submitted:
        return

    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        model_features = [s["model_key"] for s in FEATURE_SPECS]

    missing_in_ui = [f for f in model_features if f not in user_inputs]
    if missing_in_ui:
        st.error(f"UI is missing model inputs: {missing_in_ui}")
        return

    input_df = pd.DataFrame([[user_inputs[f] for f in model_features]], columns=model_features)

    try:
        risk_score = float(model.predict(input_df)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Prediction Output")
    st.metric("Predicted Risk Score", f"{risk_score:.6f}")
    render_shap_force_plot(model, model_features, input_df)


if __name__ == "__main__":
    main()
