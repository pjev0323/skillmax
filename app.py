import html as _html
import os
import re

import joblib
import nltk
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from nltk.corpus import stopwords


# -------------------------
#  PAGE CONFIGURATION AREA
# -------------------------
st.set_page_config(
    page_title="SkillMax AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def md(markup: str):
    """Render raw HTML via st.markdown.

    Strips indentation and blank lines so the markdown parser never turns
    indented HTML into code blocks.
    """
    cleaned = "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------
#  THEME LAYOUT UI GAMIT CSS AREA
# --------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* ---- Hide Streamlit chrome ---- */
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"], [data-testid="stToolbar"],
[data-testid="stStatusWidget"], #MainMenu, footer { display: none !important; }

.stApp { background: #080511 !important; color: #F8FAFC; }
[data-testid="stMain"], section.main { scroll-behavior: smooth; }
.block-container { max-width: 1280px !important; padding: 5.75rem 1.5rem 2rem !important; }
[data-testid="stVerticalBlock"] { gap: 1.5rem; }
::selection { background: #8B5CF6; color: #fff; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #080511; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(168,85,247,.6); }
.anchor-target { scroll-margin-top: 96px; }
[data-testid="stAlert"] { border-radius: 16px; }

/* ---- Fixed top navigation ---- */
.sm-nav { position: fixed; top: 0; left: 0; right: 0; z-index: 999990; backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px); background: rgba(8,5,17,.8); border-bottom: 1px solid rgba(168,85,247,.12);
  padding: 16px 24px; }
.sm-nav-inner { max-width: 1280px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; }
.sm-brand { display: flex; align-items: center; gap: 12px; }
.sm-logo { width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(to top right,#8B5CF6,#38BDF8);
  display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 20px rgba(139,92,246,.35);
  color: #fff; font-size: 18px; }
.sm-brand-name { display: block; font-size: 20px; line-height: 28px; font-weight: 800; letter-spacing: -.025em;
  background: linear-gradient(to right,#fff,#e9d5ff,#38BDF8); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent; width: fit-content; }
.sm-brand-sub { display: block; font-size: 12px; line-height: 16px; font-weight: 500; color: rgba(216,180,254,.7); margin-top: -4px; }
.sm-links { display: flex; gap: 32px; font-size: 14px; font-weight: 600; }
.sm-links a { color: #cbd5e1 !important; text-decoration: none !important; transition: color .2s; }
.sm-links a:hover { color: #38BDF8 !important; }
.sm-right { display: flex; align-items: center; gap: 16px; }
.sm-status { display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 9999px; font-size: 12px;
  font-weight: 600; background: rgba(139,92,246,.1); color: #38BDF8; border: 1px solid rgba(139,92,246,.3); }
.sm-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; margin-right: 8px; animation: smPing 2s cubic-bezier(.4,0,.6,1) infinite; }
@keyframes smPing { 50% { opacity: .5; } }
@media (max-width: 900px) { .sm-links { display: none; } }
@media (max-width: 640px) { .sm-status { display: none; } }

/* ---- Hero ---- */
.st-key-hero { position: relative; overflow: hidden; gap: 0 !important; padding: 80px 24px 64px;
  background: radial-gradient(circle at 50% 120%, rgba(139,92,246,.35) 0%, rgba(56,189,248,.15) 35%, rgba(8,5,17,0) 70%); }
.st-key-hero > div { position: relative; z-index: 2; }
.sm-globe { position: absolute; top: 0; left: 0; right: 0; margin: 0 auto; pointer-events: none; z-index: 1;
  animation: smPulse 6s ease-in-out infinite; }
@keyframes smPulse { 0%,100% { opacity: .6; transform: scale(1); } 50% { opacity: .85; transform: scale(1.02); } }
.sm-hero { text-align: center; }
.sm-badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 9999px;
  background: rgba(15,23,42,.8); border: 1px solid rgba(168,85,247,.3); color: #d8b4fe; font-size: 12px;
  line-height: 16px; font-weight: 600; margin-bottom: 24px; box-shadow: 0 10px 15px -3px rgba(0,0,0,.2);
  backdrop-filter: blur(4px); }
.sm-badge i { color: #38BDF8; }
.sm-title { font-size: 60px; line-height: 1.25; font-weight: 800; letter-spacing: -.025em; color: #fff; margin-bottom: 16px; }
.sm-title .grad { background: linear-gradient(to right,#d8b4fe,#38BDF8,#8B5CF6); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
.sm-desc { max-width: 672px; margin: 0 auto 32px; font-size: 18px; line-height: 28px; font-weight: 500; color: #94a3b8; }
@media (max-width: 768px) { .sm-title { font-size: 36px; } .sm-desc { font-size: 16px; line-height: 24px; } }

/* ---- Buttons (shared) ---- */
.stButton button, .stDownloadButton button { font-family: inherit !important; border-radius: 9999px !important;
  transition: all .25s cubic-bezier(.4,0,.2,1) !important; white-space: nowrap; line-height: 1.2; outline: none !important; }
.stButton button p, .stDownloadButton button p { margin: 0 !important; font-size: inherit !important;
  font-weight: inherit !important; color: inherit !important; }
.stButton button:disabled, .stDownloadButton button:disabled { opacity: .45; cursor: not-allowed; }

/* Reset */
.st-key-btn_reset { display: flex !important; flex-direction: column; align-items: flex-end; }
.st-key-btn_reset button { height: auto; min-height: 0; padding: 6px 16px; font-size: 12px; font-weight: 600;
  color: #d8b4fe !important; background: rgba(88,28,135,.3) !important; border: 1px solid rgba(168,85,247,.3) !important; }
.st-key-btn_reset button:hover { color: #fff !important; background: rgba(107,33,168,.4) !important; border-color: rgba(168,85,247,.3) !important; }

/* Analyze */
.st-key-btn_analyze, .st-key-btn_analyze [data-testid="stButton"] { width: 100%; }
.st-key-btn_analyze button { width: 100%; height: 48px; font-size: 14px; font-weight: 800; color: #fff !important; border: none !important;
  background: linear-gradient(to right,#8B5CF6,#9333ea,#38BDF8) !important; }
.st-key-btn_analyze button:hover { box-shadow: 0 0 25px rgba(139,92,246,.4) !important; color: #fff !important; }
.st-key-btn_analyze button:active { transform: scale(.95); }

/* Banner buttons + download */
.st-key-btn_chart button { height: 40px; padding: 0 20px; font-size: 12px; font-weight: 700; color: #fff !important;
  background: #0f172a !important; border: 1px solid rgba(168,85,247,.3) !important; width: 100%; }
.st-key-btn_chart button:hover { border-color: #c084fc !important; color: #fff !important; }
.st-key-btn_skills button, .st-key-btn_download button { height: 40px; padding: 0 20px; font-size: 12px; font-weight: 700;
  color: #fff !important; border: none !important; background: linear-gradient(to right,#8B5CF6,#38BDF8) !important;
  box-shadow: 0 4px 20px rgba(139,92,246,.35) !important; width: 100%; }
.st-key-btn_skills button:hover, .st-key-btn_download button:hover { opacity: .9; color: #fff !important; }
.st-key-btn_download { display: flex !important; flex-direction: column; align-items: flex-end; }
.st-key-btn_download [data-testid="stDownloadButton"] { width: 100%; }

/* ---- Pill tab bar ---- */
.st-key-tabbar { background: rgba(2,6,23,.8); border: 1px solid rgba(168,85,247,.2); border-radius: 9999px;
  padding: 6px; backdrop-filter: blur(12px); }
.st-key-tabbar [data-testid="stHorizontalBlock"] { gap: 8px; }
.st-key-tabbar button { width: 100%; height: 40px; padding: 0 16px; font-size: 12px; font-weight: 700; border: none !important; }
.st-key-tabbar button:is([kind="secondary"], [data-testid="stBaseButton-secondary"]) { background: transparent !important; color: #94a3b8 !important; }
.st-key-tabbar button:is([kind="secondary"], [data-testid="stBaseButton-secondary"]):hover { color: #fff !important; }
.st-key-tabbar button:is([kind="primary"], [data-testid="stBaseButton-primary"]) { color: #fff !important;
  background: linear-gradient(to right,#8B5CF6,#7e22ce) !important; box-shadow: 0 4px 20px rgba(139,92,246,.35) !important; }
.sm-qs-label { font-size: 12px; font-weight: 700; color: #d8b4fe; text-transform: uppercase; letter-spacing: .05em; text-align: right; white-space: nowrap; }

/* ---- Quick sample select ---- */
.st-key-sample_choice [data-baseweb="select"] > div { background: #0f172a !important; border: 1px solid rgba(168,85,247,.3) !important;
  border-radius: 9999px !important; min-height: 38px; }
.st-key-sample_choice [data-baseweb="select"] > div:focus-within { border-color: #38BDF8 !important; }
.st-key-sample_choice [data-baseweb="select"] div, .st-key-sample_choice [data-baseweb="select"] span { color: #e2e8f0 !important; font-size: 12px; font-weight: 500; }
.st-key-sample_choice svg { fill: #C084FC; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"] { background: #0f172a !important; }
[data-baseweb="popover"] li { color: #e2e8f0 !important; font-size: 13px; }
[data-baseweb="popover"] li:hover { background: rgba(139,92,246,.25) !important; }

/* ---- Glass panels ---- */
.glass-panel, [class*="st-key-panel_"] { background: rgba(18,12,32,.65); backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(168,85,247,.18); border-radius: 20px; padding: 24px; transition: all .3s; }
.glass-panel:hover, [class*="st-key-panel_"]:hover { border-color: rgba(192,132,252,.4); box-shadow: 0 10px 30px rgba(139,92,246,.18); }
.st-key-panel_banner { border-left: 4px solid #8B5CF6 !important; }

.sm-h3 { display: flex; align-items: center; font-size: 18px; line-height: 28px; font-weight: 700; color: #fff; }
.sm-h3 i { margin-right: 8px; }
.sm-sub { font-size: 12px; line-height: 16px; color: #94a3b8; margin-top: 6px; }
.sm-h3s { font-size: 14px; line-height: 20px; font-weight: 700; color: #d8b4fe; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 16px; }
.sm-h3s i { margin-right: 8px; }

/* Text input */
.st-key-input_widget [data-baseweb="textarea"], .st-key-input_widget [data-baseweb="base-input"], .st-key-input_widget textarea { background: rgba(2,6,23,.8) !important; }
.st-key-input_widget [data-baseweb="textarea"] { border: 1px solid rgba(168,85,247,.2) !important; border-radius: 16px !important; }
.st-key-input_widget [data-baseweb="textarea"]:focus-within { border-color: #8B5CF6 !important; box-shadow: 0 0 0 1px #8B5CF6 !important; }
.st-key-input_widget textarea { color: #e2e8f0 !important; font-size: 14px !important; line-height: 20px !important; padding: 16px !important; font-family: inherit !important; }
.st-key-input_widget textarea::placeholder { color: #475569 !important; }
.st-key-input_widget [data-testid="InputInstructions"] { color: #475569; }

/* Stats + context */
.sm-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.sm-stat { background: rgba(2,6,23,.6); padding: 16px; border-radius: 12px; border: 1px solid rgba(168,85,247,.1); text-align: center; }
.sm-stat-n { display: block; font-size: 30px; line-height: 36px; font-weight: 800; color: #fff; }
.sm-stat-l { display: block; font-size: 10px; line-height: 16px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
.sm-ctx-row { display: flex; justify-content: space-between; font-size: 12px; line-height: 16px; font-weight: 500; color: #cbd5e1;
  padding-bottom: 10px; margin-bottom: 10px; border-bottom: 1px solid rgba(30,41,59,.8); gap: 12px; }
.sm-ctx-row:last-child { border-bottom: none; padding-bottom: 0; margin-bottom: 0; }
.sm-ctx-k { color: #94a3b8; } .sm-ctx-v { font-weight: 700; color: #e9d5ff; text-align: right; }

/* Results banner */
.sm-pill { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 10px; line-height: 16px; font-weight: 800;
  text-transform: uppercase; letter-spacing: .1em; background: rgba(88,28,135,.6); color: #38BDF8; border: 1px solid rgba(168,85,247,.4); margin-bottom: 8px; }
.sm-role { font-size: 30px; line-height: 36px; font-weight: 900; color: #fff; display: flex; align-items: center; gap: 12px; }
.sm-meta { font-size: 12px; line-height: 16px; color: #94a3b8; margin-top: 4px; }

/* Ranking table */
.stMarkdown table.sm-table { width: 100%; border-collapse: collapse; font-size: 12px; color: #cbd5e1; text-align: left; margin: 0; }
.stMarkdown table.sm-table th { padding: 10px 8px !important; font-size: 10px; font-weight: 700; text-transform: uppercase;
  color: #e9d5ff; border: none !important; border-bottom: 1px solid rgba(168,85,247,.2) !important; background: transparent !important; }
.stMarkdown table.sm-table td { padding: 10px 8px !important; font-weight: 500; border: none !important;
  border-top: 1px solid rgba(30,41,59,.6) !important; background: transparent !important; }
.stMarkdown table.sm-table tbody tr:first-child td { border-top: none !important; }
.stMarkdown table.sm-table tbody tr:hover td { background: rgba(59,7,100,.2) !important; }
.stMarkdown table.sm-table .r { text-align: right; }
.stMarkdown table.sm-table .role { font-weight: 700; color: #e2e8f0; }
.stMarkdown table.sm-table .score { font-family: ui-monospace, monospace; color: #38BDF8; }
.stMarkdown table.sm-table .fit { font-family: ui-monospace, monospace; color: #d8b4fe; }
.sm-table-foot { padding-top: 16px; margin-top: 16px; border-top: 1px solid rgba(168,85,247,.1); text-align: center; font-size: 10px; color: #64748b; }
.sm-chart-empty { height: 320px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 12px; font-style: italic; }

/* Skills */
.sm-skillbox { display: flex; flex-wrap: wrap; gap: 10px; padding: 16px; background: rgba(2,6,23,.6); border-radius: 16px;
  border: 1px solid rgba(168,85,247,.15); min-height: 120px; align-items: center; }
.skill-badge { display: inline-flex; align-items: center; padding: 8px 16px; border-radius: 9999px; font-size: 12px; font-weight: 600;
  background: rgba(59,7,100,.6); color: #38BDF8; border: 1px solid rgba(168,85,247,.3); cursor: pointer;
  transition: all .25s cubic-bezier(.4,0,.2,1); }
.skill-badge i { font-size: 10px; margin-right: 6px; color: #8B5CF6; }
.skill-badge:hover { transform: translateY(-2px) scale(1.05); box-shadow: 0 0 15px rgba(56,189,248,.5); border-color: #38BDF8; color: #fff; }
.sm-empty { color: #64748b; font-size: 12px; font-style: italic; }

/* Debug */
.sm-debug-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; }
@media (max-width: 768px) { .sm-debug-grid { grid-template-columns: 1fr; } }
.sm-debug-label { display: block; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }
.sm-debug-box { background: rgba(2,6,23,.8); padding: 16px; border-radius: 12px; font-size: 12px; font-family: ui-monospace, monospace;
  height: 192px; overflow-y: auto; line-height: 1.625; white-space: pre-wrap; word-break: break-word; }

/* Footer */
.sm-footer { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; border-top: 1px solid rgba(168,85,247,.1);
  padding: 32px 0 8px; margin-top: 40px; font-size: 12px; color: #64748b; }
.sm-footer strong { color: #cbd5e1; }

/* Zero-height JS helper iframe */
[data-testid="stElementContainer"]:has(iframe[height="0"]) { position: absolute; width: 0; height: 0; overflow: hidden; margin: 0; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# -----------------------------
#  ASSETS KAG NLP HELPERS AREA
# -----------------------------
@st.cache_resource
def load_stopwords():
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
    return set(stopwords.words("english"))


@st.cache_resource
def load_assets():
    model = joblib.load("models/skillmax_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


stop_words = load_stopwords()

try:
    model, vectorizer = load_assets()
    assets_loaded = True
except Exception:
    assets_loaded = False


def clean_input_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def extract_matched_skills(cleaned_text: str, vectorizer, max_skills=25):
    vocab = vectorizer.vocabulary_
    words = cleaned_text.split()
    matched = {w for w in words if w in vocab}
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram in vocab:
            matched.add(bigram)
    return sorted(matched)[:max_skills]


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def run_analysis(text: str) -> dict:
    cleaned = clean_input_text(text)
    vec = vectorizer.transform([cleaned])
    role = model.predict(vec)[0]
    scores = model.decision_function(vec)[0]
    probs = softmax(scores)
    top = scores.argsort()[-5:][::-1]
    rankings = [
        {
            "role": str(model.classes_[i]),
            "score": float(scores[i]),
            "prob": float(probs[i] * 100),
        }
        for i in top
    ]
    return {
        "role": str(role),
        "confidence": rankings[0]["score"],
        "rankings": rankings,
        "skills": extract_matched_skills(cleaned, vectorizer, max_skills=25),
        "raw": text,
        "cleaned": cleaned,
    }


def build_report(a: dict) -> str:
    rep = (
        "=================================================\n"
        "          SKILLMAX AI ANALYSIS REPORT           \n"
        "=================================================\n\n"
        f"Predicted Primary Category : {a['role']}\n"
        f"Top Score Index            : {a['confidence']:.2f}\n\n"
        "-------------------------------------------------\n"
        "IDENTIFIED TECHNICAL SKILLS:\n"
        "-------------------------------------------------\n"
        f"{', '.join(a['skills']) if a['skills'] else 'None identified'}\n\n"
        "-------------------------------------------------\n"
        "RANKED CANDIDATE CATEGORIES:\n"
        "-------------------------------------------------\n"
    )
    rep += "\n".join(
        f"- {r['role']:<28} | Score: {r['score']:.2f} | Approx Fit: {r['prob']:.1f}%"
        for r in a["rankings"]
    )
    return rep


def build_chart(rankings: list) -> go.Figure:
    ordered = list(reversed(rankings))  # Highest score ends up on top
    fig = go.Figure(
        go.Bar(
            x=[r["score"] for r in ordered],
            y=[r["role"] for r in ordered],
            orientation="h",
            marker=dict(
                color="rgba(139,92,246,0.75)",
                line=dict(color="#A855F7", width=1.5),
            ),
            hovertemplate="%{y}<br>Decision Score: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC"),
        height=360,
        margin=dict(l=10, r=10, t=10, b=35),
        showlegend=False,
        xaxis=dict(
            gridcolor="rgba(168,85,247,0.1)",
            zeroline=False,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            showgrid=False,
            automargin=True,
            tickfont=dict(color="#F8FAFC", size=12),
        ),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#F8FAFC")),
    )
    try:  # Rounded bars require Plotly >= 6
        fig.update_traces(marker_cornerradius=20)
    except Exception:
        pass
    return fig


# -------------------------------
#  SAMPLE JOB KAG CALLBACKS AREA
# -------------------------------
SAMPLES = {
    "Data Scientist": (
        "We are seeking a Senior Data Scientist with strong Python, SQL, and Machine Learning "
        "experience. Must be proficient in pandas, scikit-learn, PyTorch, neural networks, "
        "deep learning, feature engineering, and interactive data visualization using Tableau or PowerBI."
    ),
    "DevOps Engineer": (
        "Looking for a Cloud DevOps Engineer experienced in AWS, Docker, Kubernetes, Terraform, "
        "Jenkins, CI/CD automated pipelines, bash scripting, and Linux system administration. "
        "Experience with Ansible and Prometheus is a plus."
    ),
    "Full Stack Web Developer": (
        "Hiring a Full Stack Web Developer skilled in JavaScript, TypeScript, React.js, Node.js, "
        "Express, HTML5, CSS3, RESTful APIs, GraphQL, and MongoDB database design. "
        "AWS experience preferred."
    ),
    "Cybersecurity Analyst": (
        "Seeking a Cybersecurity Analyst to monitor security posture, conduct vulnerability "
        "assessments, analyze malware, configure firewalls, manage SIEM tools, and enforce "
        "compliance frameworks like ISO27001."
    ),
}
SAMPLE_OPTIONS = [""] + list(SAMPLES.keys())
SAMPLE_LABELS = {
    "": "-- Choose Job Posting --",
    "Full Stack Web Developer": "Full Stack Developer",
}

TABS = [
    ("input", "Job Input & Analysis", ":material/search:"),
    ("analytics", "Classification Analytics", ":material/bar_chart:"),
    ("skills", "Skill Breakdown", ":material/build:"),
    ("debug", "NLP Debug", ":material/code:"),
]

st.session_state.setdefault("active_tab", "input")
st.session_state.setdefault("job_text", "")
st.session_state.setdefault("analysis", None)
st.session_state.setdefault("sample_choice", "")


def set_tab(tab_id: str):
    st.session_state["active_tab"] = tab_id


def _load_sample(key: str):
    st.session_state["job_text"] = SAMPLES[key]
    st.session_state["input_widget"] = SAMPLES[key]
    st.session_state["active_tab"] = "input"


def on_sample_select():
    key = st.session_state["sample_choice"]
    if key in SAMPLES:
        _load_sample(key)


def reset_workspace():
    st.session_state["job_text"] = ""
    st.session_state["input_widget"] = ""
    st.session_state["sample_choice"] = ""
    st.session_state["analysis"] = None


# --------------------------------------
#  TOP NAVIGATION KAG HERO SECTION AREA
# --------------------------------------
md("""
<div class="sm-nav"><div class="sm-nav-inner">
  <div class="sm-brand">
    <div class="sm-logo"><i class="fa-solid fa-bolt"></i></div>
    <div><span class="sm-brand-name">SkillMax AI</span><span class="sm-brand-sub">UNO - Recoletos College of IT</span></div>
  </div>
  <div class="sm-links">
    <a href="#overview" data-scroll="overview">DATA SCIENCE PROJECT 2026</a>
  </div>
  <div class="sm-right">
    <span class="sm-status"><span class="sm-dot"></span>LinearSVC + TF-IDF Active</span>
  </div>
</div></div>
""")

with st.container(key="hero"):
    md("""
    <div id="overview" class="anchor-target"></div>
    <div class="sm-hero">
      <div class="sm-badge"><i class="fa-solid fa-wand-magic-sparkles"></i><span>Next-Gen NLP Job Intelligence Platform</span></div>
      <div class="sm-title">Real-Time Job Analytics.<br/><span class="grad">Instant Skill Discovery.</span></div>
      <div class="sm-desc">An AI assistant designed for our Data Science project that applies NLP algorithms to evaluate job listings, predict 25 distinct IT career categories, and extract relevant skill tokens in real time.</div>
    </div>
    """)

# Animated globe + smooth-scroll helper
components.html(
    """
<script>
(function () {
  var P, doc;
  try { P = window.parent; doc = P.document; } catch (e) { return; }
  var token = {}; P.__skmToken = token;

  if (P.__skmClick) doc.removeEventListener('click', P.__skmClick, true);
  P.__skmClick = function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a[data-scroll]') : null;
    if (!a) return;
    e.preventDefault();
    var id = a.getAttribute('data-scroll');
    var t = doc.getElementById(id) || doc.getElementById('user-content-' + id);
    if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  doc.addEventListener('click', P.__skmClick, true);

  var W = 800, H = 400, N = 180, R = 160, dots = [];
  for (var i = 0; i < N; i++) {
    var th = Math.acos(2 * Math.random() - 1), ph = 2 * Math.PI * Math.random();
    dots.push({ x: R * Math.sin(th) * Math.cos(ph), y: R * Math.sin(th) * Math.sin(ph), z: R * Math.cos(th) });
  }
  var canvas = null, ctx = null, ang = 0.002;

  function ensure() {
    var hero = doc.querySelector('.st-key-hero');
    if (!hero) return false;
    if (!canvas || !canvas.isConnected || canvas.parentNode !== hero) {
      var old = hero.querySelector('canvas.sm-globe'); if (old) old.remove();
      canvas = doc.createElement('canvas');
      canvas.className = 'sm-globe'; canvas.width = W; canvas.height = H;
      hero.insertBefore(canvas, hero.firstChild);
      ctx = canvas.getContext('2d');
    }
    return true;
  }

  function frame() {
    if (P.__skmToken !== token) return;
    if (ensure()) {
      ctx.clearRect(0, 0, W, H);
      var cx = W / 2, cy = H / 2 + 30;
      ctx.fillStyle = 'rgba(168, 85, 247, 0.6)';
      for (var i = 0; i < dots.length; i++) {
        var d = dots[i];
        var x1 = d.x * Math.cos(ang) - d.z * Math.sin(ang);
        var z1 = d.z * Math.cos(ang) + d.x * Math.sin(ang);
        d.x = x1; d.z = z1;
        var s = 300 / (300 + d.z);
        if (d.z > -100) {
          ctx.beginPath();
          ctx.arc(d.x * s + cx, d.y * s + cy, 1.8 * s, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    P.requestAnimationFrame(frame);
  }
  frame();
})();
</script>
""",
    height=0,
)

if not assets_loaded:
    st.error(
        "⚠️ **Model files missing!** Please check that `models/skillmax_model.pkl` and "
        "`models/tfidf_vectorizer.pkl` exist."
    )
    st.stop()

# ----------------------------------------
#  TAB BAR KAG QUICK SAMPLE SELECTOR AREA
# ----------------------------------------
md('<div id="analyzer" class="anchor-target"></div>')

bar_col, sel_col = st.columns([3.4, 1.5], vertical_alignment="center")

with bar_col:
    with st.container(key="tabbar"):
        tab_cols = st.columns([1.2, 1.3, 0.95, 0.75])
        for col, (tab_id, label, icon) in zip(tab_cols, TABS):
            with col:
                st.button(
                    label,
                    key=f"tab_{tab_id}",
                    icon=icon,
                    type=(
                        "primary"
                        if st.session_state["active_tab"] == tab_id
                        else "secondary"
                    ),
                    on_click=set_tab,
                    args=(tab_id,),
                )

with sel_col:
    lbl_col, drop_col = st.columns([1, 1.7], vertical_alignment="center")
    with lbl_col:
        md('<div class="sm-qs-label">Quick Sample:</div>')
    with drop_col:
        st.selectbox(
            "Quick sample",
            SAMPLE_OPTIONS,
            key="sample_choice",
            format_func=lambda k: SAMPLE_LABELS.get(k, k),
            on_change=on_sample_select,
            label_visibility="collapsed",
        )

active = st.session_state["active_tab"]

# -------------------------------
#  JOB INPUT & ANALYSIS TAB AREA
# -------------------------------
if active == "input":
    left, right = st.columns([2, 1], gap="medium")

    with left:
        with st.container(key="panel_input"):
            head_col, reset_col = st.columns([5, 1], vertical_alignment="center")
            with head_col:
                md(
                    '<div class="sm-h3"><i class="fa-regular fa-file-lines"'
                    ' style="color:#38BDF8"></i>Paste Job Posting Text</div>'
                )
            with reset_col:
                st.button(
                    "Reset",
                    key="btn_reset",
                    icon=":material/restart_alt:",
                    on_click=reset_workspace,
                )

            if "input_widget" not in st.session_state:
                st.session_state["input_widget"] = st.session_state["job_text"]

            job_text = st.text_area(
                "Job Posting Text",
                key="input_widget",
                height=240,
                placeholder=(
                    "Paste full IT job posting text here (including duties, qualifications, "
                    "and technology stack requirements)..."
                ),
                label_visibility="collapsed",
            )
            st.session_state["job_text"] = job_text
            analyze_clicked = st.button(
                "Analyze Job Posting", key="btn_analyze", icon=":material/bolt:"
            )

    with right:
        stripped = job_text.strip()
        words = len(stripped.split()) if stripped else 0
        chars = len(stripped)

        md(f"""
        <div class="glass-panel">
          <div class="sm-h3s"><i class="fa-solid fa-chart-simple"></i>Text Metrics</div>
          <div class="sm-stat-grid">
            <div class="sm-stat"><span class="sm-stat-n">{words}</span><span class="sm-stat-l">Word Count</span></div>
            <div class="sm-stat"><span class="sm-stat-n">{chars}</span><span class="sm-stat-l">Characters</span></div>
          </div>
        </div>
        """)
        md("""
        <div class="glass-panel">
          <div class="sm-h3s" style="margin-bottom:12px"><i class="fa-solid fa-circle-info"></i>System Context</div>
          <div class="sm-ctx-row"><span class="sm-ctx-k">Institution:</span><span class="sm-ctx-v">UNO - Recoletos</span></div>
          <div class="sm-ctx-row"><span class="sm-ctx-k">Classifier:</span><span class="sm-ctx-v" style="color:#38BDF8">Linear SVC</span></div>
          <div class="sm-ctx-row"><span class="sm-ctx-k">Vectorization:</span><span class="sm-ctx-v">TF-IDF (Unigram + Bigram)</span></div>
          <div class="sm-ctx-row"><span class="sm-ctx-k">Coverage Scope:</span><span class="sm-ctx-v" style="color:#34d399">25 IT Job Categories</span></div>
        </div>
        """)

    if analyze_clicked:
        if not job_text.strip():
            st.warning("Please paste or select a job description text to analyze.")
        else:
            with st.spinner("Executing NLP pre-processing & ML inference..."):
                st.session_state["analysis"] = run_analysis(job_text.strip())

    analysis = st.session_state["analysis"]
    if analysis:
        with st.container(key="panel_banner"):
            b_text, b_chart, b_skills = st.columns(
                [5, 1.25, 1.25], vertical_alignment="center"
            )
            with b_text:
                md(f"""
                <div>
                  <span class="sm-pill">Primary Predicted Category</span>
                  <div class="sm-role">👨‍💻 <span>{_html.escape(analysis['role'])}</span></div>
                  <div class="sm-meta">Top Score Index: <strong style="color:#d8b4fe">{analysis['confidence']:.2f}</strong>
                  | Extracted Skill Tokens: <strong style="color:#38BDF8">{len(analysis['skills'])} phrases</strong></div>
                </div>
                """)
            with b_chart:
                st.button(
                    "View Full Chart",
                    key="btn_chart",
                    on_click=set_tab,
                    args=("analytics",),
                )
            with b_skills:
                st.button(
                    "Explore Skills",
                    key="btn_skills",
                    on_click=set_tab,
                    args=("skills",),
                )

# -----------------------------------
#  CLASSIFICATION ANALYTICS TAB AREA
# -----------------------------------
elif active == "analytics":
    analysis = st.session_state["analysis"]
    left, right = st.columns([2, 1], gap="medium")

    with left:
        with st.container(key="panel_chart"):
            md("""
            <div>
              <div class="sm-h3"><i class="fa-solid fa-chart-column" style="color:#8B5CF6"></i>Model Category Confidence Breakdown</div>
              <div class="sm-sub">Top decision scores computed across candidate role vectors.</div>
            </div>
            """)
            if analysis:
                st.plotly_chart(
                    build_chart(analysis["rankings"]),
                    theme=None,
                    config={"displayModeBar": False},
                )
            else:
                md('<div class="sm-chart-empty">Run an analysis to view the confidence chart.</div>')

    with right:
        if analysis:
            rows = "".join(
                f'<tr><td class="role">{_html.escape(r["role"])}</td>'
                f'<td class="r score">{r["score"]:.2f}</td>'
                f'<td class="r fit">{r["prob"]:.1f}%</td></tr>'
                for r in analysis["rankings"]
            )
        else:
            rows = '<tr><td colspan="3" style="text-align:center;color:#64748b">Run an analysis to view candidate rankings.</td></tr>'

        md(f"""
        <div class="glass-panel">
          <div class="sm-h3s"><i class="fa-solid fa-table-list"></i>Ranked Candidate Scores</div>
          <div style="overflow-x:auto">
            <table class="sm-table">
              <thead><tr><th>Job Category</th><th class="r">Score</th><th class="r">Fit %</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
          <div class="sm-table-foot">Calibrated Softmax Probability &amp; Linear Decision Functions</div>
        </div>
        """)

# -----------------------------------------
#  SKILL BREAKDOWN & EXPORT SKILL TAB AREA
# -----------------------------------------
elif active == "skills":
    analysis = st.session_state["analysis"]
    with st.container(key="panel_skills"):
        t_col, d_col = st.columns([3, 1.3], vertical_alignment="center")
        with t_col:
            md("""
            <div>
              <div class="sm-h3"><i class="fa-solid fa-tags" style="color:#38BDF8"></i>Identified Technical Stack Tokens</div>
              <div class="sm-sub">Extracted vocabulary keywords and bi-gram technology phrases matched against TF-IDF index.</div>
            </div>
            """)
        with d_col:
            file_name = (
                f"skillmax_{analysis['role'].lower().replace(' ', '_')}_analysis.txt"
                if analysis
                else "skillmax_analysis.txt"
            )
            st.download_button(
                "Download Analysis (.txt)",
                data=build_report(analysis) if analysis else "",
                file_name=file_name,
                mime="text/plain",
                key="btn_download",
                icon=":material/download:",
                disabled=analysis is None,
            )

        if analysis and analysis["skills"]:
            pills = "".join(
                f'<span class="skill-badge"><i class="fa-solid fa-bolt"></i>{_html.escape(s)}</span>'
                for s in analysis["skills"]
            )
        elif analysis:
            pills = '<span class="sm-empty">No specific technical terms identified.</span>'
        else:
            pills = '<span class="sm-empty">No skills extracted yet. Paste a job description and click Analyze.</span>'

        md(f'<div class="sm-skillbox">{pills}</div>')

# --------------------
#  NLP DEBUG TAB AREA
# --------------------
elif active == "debug":
    analysis = st.session_state["analysis"]
    raw = _html.escape(analysis["raw"]) if analysis else "No text loaded."
    cleaned = (
        _html.escape(analysis["cleaned"])
        if analysis and analysis["cleaned"]
        else "No tokens generated."
    )

    md(f"""
    <div class="glass-panel">
      <div class="sm-h3"><i class="fa-solid fa-bug" style="color:#c084fc"></i>NLP Normalization &amp; Debug Log</div>
      <div class="sm-sub">Tokenization, stopword removal, regex HTML cleaning, and lemmatization pipeline inspect.</div>
      <div class="sm-debug-grid">
        <div>
          <span class="sm-debug-label" style="color:#d8b4fe">Raw Text Snippet</span>
          <div class="sm-debug-box" style="border:1px solid rgba(168,85,247,.2);color:#94a3b8">{raw}</div>
        </div>
        <div>
          <span class="sm-debug-label" style="color:#34d399">Cleaned &amp; Tokenized Stream</span>
          <div class="sm-debug-box" style="border:1px solid rgba(16,185,129,.2);color:rgba(110,231,183,.8)">{cleaned}</div>
        </div>
      </div>
    </div>
    """)

# ------------
# FOOTER AREA
# ------------
md("""
<div id="metadata" class="sm-footer anchor-target">
  <div>⚡ SkillMax AI Assistant &bull; Developed by <strong>Group DATA-MAX</strong></div>
  <div>College of Information Technology &bull; University of Negros Occidental - Recoletos</div>
</div>
""")
