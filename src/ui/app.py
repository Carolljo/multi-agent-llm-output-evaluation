import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Multi-Agent LLM Output Evaluation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background: #0b0f14;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 0 2rem 0;
            border-bottom: 1px solid #27303a;
            margin-bottom: 2rem;
        }

        .eyebrow {
            color: #8b98a7;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .hero-title {
            color: #f1f5f9;
            font-size: 2.4rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0;
        }

        .hero-subtitle {
            color: #8b98a7;
            font-size: 1rem;
            margin-top: 0.65rem;
        }

        .section-label {
            color: #8b98a7;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .pipeline {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 1rem 0 2rem 0;
            overflow-x: auto;
        }

        .pipeline-step {
            background: #121820;
            border: 1px solid #27303a;
            border-radius: 8px;
            padding: 0.65rem 0.9rem;
            color: #cbd5e1;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .pipeline-arrow {
            color: #586575;
            font-size: 0.9rem;
        }

        div[data-testid="stTextArea"] textarea {
            background: #10161d;
            color: #e5e7eb;
            border: 1px solid #303b47;
            border-radius: 8px;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #64748b;
            box-shadow: none;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #64748b;
            background: #e5e7eb;
            color: #0b0f14;
            font-weight: 700;
            padding: 0.65rem 1rem;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: #ffffff;
            border-color: #94a3b8;
            color: #0b0f14;
        }

        .status-card {
            background: #10161d;
            border: 1px solid #27303a;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .status-label {
            color: #7f8b99;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .status-value {
            color: #f1f5f9;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .divider {
            height: 1px;
            background: #27303a;
            margin: 2rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI Quality Evaluation System</div>
        <div class="hero-title">Multi-Agent LLM Output Evaluation</div>
        <div class="hero-subtitle">
            Independent evaluation, disagreement detection, and final adjudication.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Pipeline visualization
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-label">Evaluation Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pipeline">
        <div class="pipeline-step">Candidate Output</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Accuracy</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Logic</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Completeness</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Aggregation</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Disagreement</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Adjudication</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Final Result</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Input workspace
# -------------------------------------------------------------------

st.markdown(
    '<div class="section-label">Evaluation Input</div>',
    unsafe_allow_html=True,
)

with st.form("evaluation_form"):

    question = st.text_area(
        "Question",
        placeholder="Enter the question being evaluated...",
        height=120,
    )

    response = st.text_area(
        "Candidate Response",
        placeholder="Paste the LLM-generated response...",
        height=220,
    )

    reference_answer = st.text_area(
        "Reference Answer",
        placeholder="Enter the expected or reference answer...",
        height=180,
    )

    submitted = st.form_submit_button(
        "Run Multi-Agent Evaluation",
        type="primary",
    )


# -------------------------------------------------------------------
# API call
# -------------------------------------------------------------------

if submitted:

    if not question.strip():
        st.error("Question cannot be empty.")
        st.stop()

    if not response.strip():
        st.error("Candidate response cannot be empty.")
        st.stop()

    if not reference_answer.strip():
        st.error("Reference answer cannot be empty.")
        st.stop()

    payload = {
        "question": question,
        "response": response,
        "reference_answer": reference_answer,
    }

    try:
        with st.spinner("Running multi-agent evaluation..."):

            api_response = requests.post(
                f"{API_URL}/evaluate",
                json=payload,
                timeout=300,
            )

        if api_response.status_code != 200:
            st.error(
                f"Evaluation API returned HTTP {api_response.status_code}."
            )
            st.stop()

        result = api_response.json()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-label">Evaluation Complete</div>',
            unsafe_allow_html=True,
        )

        final_result = result["final_result"]
        disagreement = result["disagreement"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Final Verdict",
                final_result["verdict"],
            )

        with col2:
            st.metric(
                "Final Score",
                f'{final_result["score"]:.1f} / 10',
            )

        with col3:
            st.metric(
                "Confidence",
                f'{final_result["confidence"]:.0%}',
            )

        st.markdown(
            '<div class="divider"></div>',
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # Evaluator matrix
        # ------------------------------------------------------------

        st.markdown(
            '<div class="section-label">Independent Evaluators</div>',
            unsafe_allow_html=True,
        )

        evaluator_columns = st.columns(3)

        for column, evaluation in zip(
            evaluator_columns,
            result["evaluations"],
        ):
            with column:

                st.markdown(
                    f"""
                    <div class="status-card">
                        <div class="status-label">
                            {evaluation["criterion"]}
                        </div>
                        <div class="status-value">
                            {evaluation["score"]} / 10
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    evaluation["score"] / 10,
                    text=f'Confidence: {evaluation["confidence"]:.0%}',
                )

                with st.expander("Reasoning"):
                    st.write(evaluation["reasoning"])

                if evaluation["issues"]:
                    with st.expander("Issues"):
                        for issue in evaluation["issues"]:
                            st.warning(issue)
                else:
                    st.caption("No issues identified.")

        # ------------------------------------------------------------
        # Disagreement
        # ------------------------------------------------------------

        st.markdown(
            '<div class="divider"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-label">Disagreement Analysis</div>',
            unsafe_allow_html=True,
        )

        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            status = (
                "Detected"
                if disagreement["has_disagreement"]
                else "No disagreement"
            )
            st.metric("Status", status)

        with d_col2:
            st.metric(
                "Severity",
                disagreement["severity"].upper(),
            )

        with d_col3:
            st.metric(
                "Score Spread",
                disagreement["score_spread"],
            )

        if disagreement["reasons"]:
            with st.expander("Disagreement Reasons"):
                for reason in disagreement["reasons"]:
                    st.write(f"• {reason}")

        # ------------------------------------------------------------
        # Adjudication
        # ------------------------------------------------------------

        adjudication = result["adjudication"]

        if adjudication is not None:

            st.markdown(
                '<div class="divider"></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-label">Final Adjudication</div>',
                unsafe_allow_html=True,
            )

            a_col1, a_col2, a_col3 = st.columns(3)

            with a_col1:
                st.metric(
                    "Adjudicator Verdict",
                    adjudication["final_verdict"],
                )

            with a_col2:
                st.metric(
                    "Adjudicator Score",
                    f'{adjudication["final_score"]} / 10',
                )

            with a_col3:
                st.metric(
                    "Confidence",
                    f'{adjudication["confidence"]:.0%}',
                )

            with st.expander("Adjudicator Reasoning", expanded=True):
                st.write(adjudication["reasoning"])

            if adjudication["issues"]:
                with st.expander("Adjudicator Issues", expanded=True):
                    for issue in adjudication["issues"]:
                        st.warning(issue)

        # ------------------------------------------------------------
        # Final reasoning
        # ------------------------------------------------------------

        st.markdown(
            '<div class="divider"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-label">Final Reasoning</div>',
            unsafe_allow_html=True,
        )

        st.write(final_result["reasoning"])

        if final_result["issues"]:
            st.markdown(
                '<div class="section-label">Final Issues</div>',
                unsafe_allow_html=True,
            )

            for issue in final_result["issues"]:
                st.warning(issue)

    except requests.RequestException:
        st.error(
            "Could not connect to the evaluation API. "
            "Make sure the FastAPI server is running."
        )