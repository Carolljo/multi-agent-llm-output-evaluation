import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Multi-Agent Evaluation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(56,189,248,.08), transparent 25%),
            radial-gradient(circle at 10% 30%, rgba(139,92,246,.07), transparent 28%),
            #080b10;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1500px;
        padding: 2.2rem 3rem 4rem;
    }

    .topline {
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:2.2rem;
    }

    .brand {
        display:flex;
        align-items:center;
        gap:.75rem;
    }

    .brand-mark {
        width:38px;
        height:38px;
        border:1px solid #334155;
        border-radius:10px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:1.15rem;
        background:#0d131b;
        color:#cbd5e1;
    }

    .brand-name {
        font-size:.82rem;
        font-weight:800;
        letter-spacing:.14em;
        text-transform:uppercase;
        color:#cbd5e1;
    }

    .system-pill {
        border:1px solid #263241;
        background:#0d131b;
        border-radius:999px;
        padding:.38rem .7rem;
        font-size:.7rem;
        letter-spacing:.08em;
        text-transform:uppercase;
        color:#94a3b8;
    }

    .hero {
        margin-bottom:2.4rem;
    }

    .eyebrow {
        color:#64748b;
        font-size:.72rem;
        font-weight:800;
        letter-spacing:.18em;
        text-transform:uppercase;
        margin-bottom:.65rem;
    }

    .hero h1 {
        margin:0;
        font-size:3rem;
        line-height:1;
        letter-spacing:-.045em;
        color:#f8fafc;
    }

    .hero p {
        color:#7f8ea3;
        margin:.85rem 0 0;
        font-size:1rem;
        max-width:700px;
    }

    .section {
        margin:2.1rem 0 .85rem;
        display:flex;
        align-items:center;
        gap:.8rem;
    }

    .section span {
        color:#64748b;
        font-size:.68rem;
        font-weight:800;
        letter-spacing:.16em;
        text-transform:uppercase;
    }

    .section-line {
        height:1px;
        background:#1d2733;
        flex:1;
    }

    .workspace {
        background:#0b1017;
        border:1px solid #1f2a37;
        border-radius:16px;
        padding:1.25rem;
    }

    div[data-testid="stTextArea"] label {
        color:#94a3b8 !important;
        font-size:.72rem !important;
        font-weight:800 !important;
        letter-spacing:.08em;
        text-transform:uppercase;
    }

    div[data-testid="stTextArea"] textarea {
        background:#080c12 !important;
        color:#e2e8f0 !important;
        border:1px solid #263241 !important;
        border-radius:10px !important;
        line-height:1.55 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color:#475569 !important;
        box-shadow:0 0 0 1px #334155 !important;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        width:100%;
        min-height:48px;
        border-radius:10px;
        border:1px solid #475569;
        background:#e2e8f0;
        color:#080b10;
        font-weight:900;
        letter-spacing:.04em;
        text-transform:uppercase;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background:#f8fafc;
        border-color:#94a3b8;
    }

    .decision {
        border:1px solid #334155;
        border-radius:18px;
        padding:1.55rem 1.7rem;
        background:
            linear-gradient(135deg, rgba(30,41,59,.45), rgba(10,14,20,.9));
        margin-bottom:1rem;
    }

    .decision-kicker {
        color:#64748b;
        font-size:.68rem;
        font-weight:900;
        letter-spacing:.16em;
        text-transform:uppercase;
    }

    .decision-main {
        display:flex;
        justify-content:space-between;
        align-items:end;
        gap:1rem;
        margin-top:.45rem;
    }

    .verdict {
        color:#f8fafc;
        font-size:2.35rem;
        font-weight:900;
        letter-spacing:-.04em;
    }

    .score {
        color:#cbd5e1;
        font-size:1.05rem;
        font-weight:700;
    }

    .decision-meta {
        margin-top:1rem;
        padding-top:1rem;
        border-top:1px solid #263241;
        color:#94a3b8;
        font-size:.78rem;
    }

    .node {
        border:1px solid #253140;
        background:#0b1017;
        border-radius:12px;
        padding:1rem;
        min-height:100px;
    }

    .node-role {
        color:#64748b;
        font-size:.65rem;
        font-weight:900;
        letter-spacing:.14em;
        text-transform:uppercase;
    }

    .node-title {
        color:#f1f5f9;
        font-size:1rem;
        font-weight:800;
        margin-top:.3rem;
    }

    .node-score {
        color:#cbd5e1;
        font-size:1.45rem;
        font-weight:900;
        margin-top:.55rem;
    }

    .node-confidence {
        color:#64748b;
        font-size:.7rem;
        margin-top:.1rem;
    }

    .flow {
        text-align:center;
        color:#475569;
        font-size:1.15rem;
        padding-top:1.8rem;
    }

    .signal {
        border-radius:12px;
        border:1px solid #293646;
        background:#0c121a;
        padding:1rem 1.1rem;
        margin-top:1rem;
    }

    .signal-label {
        color:#64748b;
        font-size:.65rem;
        font-weight:900;
        letter-spacing:.14em;
        text-transform:uppercase;
    }

    .signal-value {
        color:#e2e8f0;
        font-size:1.1rem;
        font-weight:900;
        margin-top:.25rem;
    }

    .reasoning {
        border-left:2px solid #334155;
        padding:.2rem 0 .2rem 1rem;
        color:#cbd5e1;
        line-height:1.65;
        font-size:.92rem;
    }

    .issue {
        border:1px solid #3a3030;
        background:#140f11;
        border-radius:9px;
        padding:.75rem .9rem;
        margin:.45rem 0;
        color:#d6c6c8;
        font-size:.84rem;
    }

    .adjudication {
        border:1px solid #4b5563;
        background:#10151c;
        border-radius:14px;
        padding:1.25rem;
    }

    .adjudication-title {
        color:#f8fafc;
        font-weight:900;
        font-size:1.05rem;
    }

    .adjudication-sub {
        color:#64748b;
        font-size:.7rem;
        letter-spacing:.1em;
        text-transform:uppercase;
        margin-top:.2rem;
    }

    .stProgress > div > div > div {
        background:#cbd5e1;
    }

    .footer {
        margin-top:3rem;
        padding-top:1rem;
        border-top:1px solid #1d2733;
        color:#475569;
        font-size:.7rem;
        text-align:right;
        letter-spacing:.06em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str):
    st.markdown(
        f"""
        <div class="section">
            <span>{title}</span>
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def issue_list(items):
    for item in items:
        st.markdown(f'<div class="issue">{item}</div>', unsafe_allow_html=True)


# Header
st.markdown(
    """
    <div class="topline">
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div class="brand-name">Multi-Agent Evaluation</div>
        </div>
        <div class="system-pill">Local Evaluation Engine</div>
    </div>

    <div class="hero">
        <div class="eyebrow">AI Output Reliability Control Center</div>
        <h1>Evaluate. Detect. Arbitrate.</h1>
        <p>
            Independent evaluators inspect factual accuracy, logical validity,
            and completeness before disagreement is resolved by an adjudicator.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input workspace
section("Evaluation Workspace")

with st.form("evaluation_form"):
    st.markdown('<div class="workspace">', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        question = st.text_area(
            "Question",
            placeholder="What should the model answer?",
            height=180,
        )
        reference_answer = st.text_area(
            "Reference Answer",
            placeholder="What constitutes the expected answer?",
            height=210,
        )

    with right:
        response = st.text_area(
            "Candidate Response",
            placeholder="Paste the LLM-generated response here...",
            height=420,
        )

    submitted = st.form_submit_button(
        "Run Multi-Agent Evaluation",
        type="primary",
    )

    st.markdown("</div>", unsafe_allow_html=True)


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
        with st.spinner("Running independent evaluators and arbitration..."):
            api_response = requests.post(
                f"{API_URL}/evaluate",
                json=payload,
                timeout=300,
            )

        if api_response.status_code != 200:
            st.error(f"Evaluation API returned HTTP {api_response.status_code}.")
            st.stop()

        result = api_response.json()

    except requests.RequestException:
        st.error(
            "Could not connect to the evaluation API. "
            "Make sure FastAPI is running on 127.0.0.1:8000."
        )
        st.stop()

    final_result = result["final_result"]
    disagreement = result["disagreement"]
    evaluations = result["evaluations"]
    adjudication = result.get("adjudication")

    # Final decision
    section("Final Decision")

    st.markdown(
        f"""
        <div class="decision">
            <div class="decision-kicker">System Verdict</div>
            <div class="decision-main">
                <div class="verdict">{final_result["verdict"]}</div>
                <div class="score">{final_result["score"]:.1f} / 10</div>
            </div>
            <div class="decision-meta">
                Confidence: {final_result["confidence"]:.0%}
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Disagreement: {"Detected" if disagreement["has_disagreement"] else "None"}
                &nbsp;&nbsp;·&nbsp;&nbsp;
                Adjudication: {"Triggered" if adjudication else "Not required"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Evaluator network
    section("Independent Evaluator Network")

    cols = st.columns(3, gap="medium")

    for col, evaluation in zip(cols, evaluations):
        with col:
            criterion = evaluation["criterion"].upper()
            st.markdown(
                f"""
                <div class="node">
                    <div class="node-role">Independent Agent</div>
                    <div class="node-title">{criterion}</div>
                    <div class="node-score">{evaluation["score"]} / 10</div>
                    <div class="node-confidence">
                        Confidence {evaluation["confidence"]:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                min(max(evaluation["score"] / 10, 0), 1),
            )

            with st.expander("Agent reasoning"):
                st.markdown(
                    f'<div class="reasoning">{evaluation["reasoning"]}</div>',
                    unsafe_allow_html=True,
                )

            if evaluation["issues"]:
                with st.expander("Detected issues"):
                    issue_list(evaluation["issues"])

    # Decision flow
    section("Consensus & Arbitration")

    flow1, arrow, flow2 = st.columns([1, .25, 1])

    with flow1:
        st.markdown(
            f"""
            <div class="signal">
                <div class="signal-label">Consensus Engine</div>
                <div class="signal-value">
                    {"Disagreement detected" if disagreement["has_disagreement"] else "Evaluator agreement"}
                </div>
                <div class="node-confidence">
                    Severity: {disagreement["severity"].upper()}
                    &nbsp;·&nbsp;
                    Spread: {disagreement["score_spread"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with arrow:
        st.markdown('<div class="flow">→</div>', unsafe_allow_html=True)

    with flow2:
        st.markdown(
            f"""
            <div class="signal">
                <div class="signal-label">Resolution Layer</div>
                <div class="signal-value">
                    {"Adjudicator engaged" if adjudication else "Direct finalization"}
                </div>
                <div class="node-confidence">
                    {"Independent final judgment required" if adjudication else "Evaluator consensus sufficient"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if disagreement["reasons"]:
        with st.expander("Why the evaluators disagreed"):
            issue_list(disagreement["reasons"])

    # Adjudicator
    if adjudication:
        section("Adjudication")

        st.markdown(
            f"""
            <div class="adjudication">
                <div class="adjudication-title">
                    Final arbitration by adjudicator
                </div>
                <div class="adjudication-sub">
                    Conflict resolution layer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        a1, a2, a3 = st.columns(3)

        with a1:
            st.metric("Verdict", adjudication["final_verdict"])

        with a2:
            st.metric("Score", f'{adjudication["final_score"]} / 10')

        with a3:
            st.metric("Confidence", f'{adjudication["confidence"]:.0%}')

        with st.expander("Adjudicator reasoning", expanded=True):
            st.markdown(
                f'<div class="reasoning">{adjudication["reasoning"]}</div>',
                unsafe_allow_html=True,
            )

        if adjudication["issues"]:
            with st.expander("Adjudicator issues", expanded=True):
                issue_list(adjudication["issues"])

    # Final reasoning
    section("Decision Rationale")

    st.markdown(
        f'<div class="reasoning">{final_result["reasoning"]}</div>',
        unsafe_allow_html=True,
    )

    if final_result["issues"]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="eyebrow">Final Issues</div>',
            unsafe_allow_html=True,
        )
        issue_list(final_result["issues"])

    st.markdown(
        """
        <div class="footer">
            MULTI-AGENT LLM OUTPUT EVALUATION · EVALUATION ENGINE v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )
