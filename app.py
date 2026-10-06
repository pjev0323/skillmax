import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# 1. Page Configuration & Full UI Overrides
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkillMax AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Cosmic Theme & Header Styling (Exact match to Preview HTML)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hide Streamlit Default Top Header & Adjust Padding */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px !important;
    }

    /* Background Canvas */
    .stApp {
        background-color: #080511 !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.3) 0%, transparent 60%),
            radial-gradient(circle at 80% 80%, rgba(56, 189, 248, 0.1) 0%, transparent 40%);
        color: #F8FAFC;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0C0716 !important;
        border-right: 1px solid rgba(168, 85, 247, 0.18) !important;
    }

    /* Custom Top Navigation Header Bar */
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background: rgba(8, 5, 17, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 50px;
        margin-bottom: 30px;
    }
    
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #8B5CF6, #38BDF8);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        background: linear-gradient(120deg, #FFFFFF, #E9D5FF, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub {
        font-size: 0.7rem;
        color: rgba(216, 180, 254, 0.7);
        margin-top: -3px;
    }

    /* Hero Section Header (Centered) */
    .hero-center {
        text-align: center;
        max-width: 850px;
        margin: 0 auto 15px auto;
        padding: 10px 20px;
    }

    .hero-badge-pill {
        display: inline-block;
        background: rgba(168, 85, 247, 0.12);
        border: 1px solid rgba(168, 85, 247, 0.3);
        color: #C084FC;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 18px;
    }

    .hero-title-main {
        font-size: 3.2rem !important;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -1px;
        color: #FFFFFF;
        margin-bottom: 16px;
    }

    .hero-gradient-text {
        background: linear-gradient(120deg, #D8B4FE 0%, #38BDF8 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 10px;
        line-height: 1.6;
    }

    /* Glassmorphic Interactive Cards */
    .glass-card {
        background: rgba(18, 12, 32, 0.65);
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        border-color: rgba(192, 132, 252, 0.4);
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.18);
    }

    /* Buttons Override */
    div.stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #38BDF8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 28px rgba(56, 189, 248, 0.65) !important;
        transform: translateY(-2px) scale(1.01) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #C084FC !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background: rgba(168, 85, 247, 0.2) !important;
        color: #F8FAFC !important;
        border-color: #C084FC !important;
        box-shadow: 0 0 18px rgba(168, 85, 247, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* Skill Tag Badges */
    .skill-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(139, 92, 246, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(168, 85, 247, 0.35);
        padding: 7px 16px;
        margin: 5px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        transition: all 0.25s ease;
        cursor: pointer;
    }

    .skill-pill:hover {
        background: rgba(56, 189, 248, 0.2);
        border-color: #38BDF8;
        color: #FFFFFF;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
        transform: translateY(-2px) scale(1.03);
    }

    /* Workspace Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(12, 7, 22, 0.85);
        padding: 6px;
        border-radius: 30px;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 20px;
        padding: 8px 22px;
        color: #94A3B8;
        font-weight: 600;
        transition: all 0.25s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #38BDF8 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }

    /* Quick Stat Cards */
    .stat-box {
        background: rgba(12, 7, 22, 0.85);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(168, 85, 247, 0.18);
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-box:hover {
        border-color: rgba(168, 85, 247, 0.45);
        background: rgba(18, 12, 32, 0.8);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #A855F7;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Prediction Category Card */
    .prediction-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8B5CF6 0%, #38BDF8 100%);
        color: #FFFFFF;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        margin-bottom: 12px;
    }

    .predicted-role-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Asset Loading & Preprocessing
# -----------------------------------------------------------------------------
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_assets():
    model = joblib.load("models/skillmax_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_assets()
    assets_loaded = True
except Exception:
    assets_loaded = False

def clean_input_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def extract_matched_skills(cleaned_text, vectorizer, max_skills=25):
    vocab = vectorizer.vocabulary_
    words = cleaned_text.split()
    matched = set()
    
    for word in words:
        if word in vocab:
            matched.add(word)
            
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in vocab:
            matched.add(bigram)
            
    return sorted(list(matched))[:max_skills]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

SAMPLE_DESCRIPTIONS = {
    "-- Select an Interactive Sample --": "",
    "Data Scientist": "We are seeking a Senior Data Scientist with strong Python, SQL, and Machine Learning experience. Must be proficient in pandas, scikit-learn, PyTorch, neural networks, deep learning, feature engineering, and interactive data visualization using Tableau or PowerBI.",
    "DevOps Engineer": "Looking for a Cloud DevOps Engineer experienced in AWS, Docker, Kubernetes, Terraform, Jenkins, CI/CD automated pipelines, bash scripting, and Linux system administration. Experience with Ansible and Prometheus is a plus.",
    "Full Stack Web Developer": "Hiring a Full Stack Web Developer skilled in JavaScript, TypeScript, React.js, Node.js, Express, HTML5, CSS3, RESTful APIs, GraphQL, and MongoDB database design. AWS experience preferred.",
    "Cybersecurity Analyst": "Seeking a Cybersecurity Analyst to monitor security posture, conduct vulnerability assessments, analyze malware, configure firewalls, manage SIEM tools, and enforce compliance frameworks like ISO27001."
}

LOGO_PATH = "assets/unor_logo.png"

# Initialize Session State
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False

# Callbacks
def on_sample_select():
    selected = st.session_state["sample_choice"]
    if selected != "-- Select an Interactive Sample --":
        st.session_state["input_text"] = SAMPLE_DESCRIPTIONS[selected]
    st.session_state["analyzed"] = False

def reset_workspace_callback():
    st.session_state["input_text"] = ""
    st.session_state["analyzed"] = False
    st.session_state["sample_choice"] = "-- Select an Interactive Sample --"

# -----------------------------------------------------------------------------
# 3. Sidebar UI Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<h3 style='color:#C084FC;'>🏛 UNO - Recoletos</h3>", unsafe_allow_html=True)
        st.caption("📍 *Place logo in assets/unor_logo.png*")

    st.markdown("---")
    st.markdown("<h4 style='color:#F8FAFC;'>⚡ SkillMax AI</h4>", unsafe_allow_html=True)
    st.write(
        "Automated IT job posting classification & skill extraction powered by "
        "Natural Language Processing (NLP)."
    )

    st.markdown("---")
    st.markdown("### 🧪 Quick Load Templates")
    st.selectbox(
        "Choose a pre-filled job posting:",
        list(SAMPLE_DESCRIPTIONS.keys()),
        key="sample_choice",
        on_change=on_sample_select
    )

    st.markdown("---")
    st.markdown("### 📋 Project Metadata")
    st.markdown("""
    - **Institution:** UNO - Recoletos
    - **Department:** College of IT
    - **Classifier:** Linear SVC
    - **Vectorization:** TF-IDF (Unigram + Bigram)
    - **Scope:** 25 IT Categories
    """)
    st.markdown("---")
    st.caption("Developed by **Group DATA-MAX**")

# -----------------------------------------------------------------------------
# 4. Top Navigation Bar (Matching Preview Header)
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="top-nav-bar">
        <div class="brand-logo">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-title">SkillMax AI</div>
                <div class="brand-sub">UNO - Recoletos College of IT</div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span style="font-size: 0.75rem; font-weight: 700; color: #38BDF8; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); padding: 4px 14px; border-radius: 20px;">
                🟢 LinearSVC + TF-IDF Active
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Centered Hero Section (Without Action Buttons)
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-center">
        <span class="hero-badge-pill">✨ Next-Gen NLP Job Intelligence Platform</span>
        <h1 class="hero-title-main">
            Real-Time Job Analytics. <br/>
            <span class="hero-gradient-text">Instant Skill Discovery.</span>
        </h1>
        <p class="hero-desc">
            Categorize unstructured IT job listings across 25 target technical roles and extract candidate skills in milliseconds using Machine Learning.
        </p>
    </div>
""", unsafe_allow_html=True)

# Ambient Canvas Mesh Sphere
components.html("""
    <canvas id="globeCanvas" style="width: 100%; height: 160px;"></canvas>
    <script>
        const canvas = document.getElementById('globeCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = 160;

        let dots = [];
        const dotCount = 180;
        const radius = 65;

        for (let i = 0; i < dotCount; i++) {
            let theta = Math.acos(2 * Math.random() - 1);
            let phi = 2 * Math.PI * Math.random();
            dots.push({
                x: radius * Math.sin(theta) * Math.cos(phi),
                y: radius * Math.sin(theta) * Math.sin(phi),
                z: radius * Math.cos(theta)
            });
        }

        let angleY = 0.008;

        function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            let cx = canvas.width / 2;
            let cy = canvas.height / 2;

            for (let i = 0; i < dots.length; i++) {
                let d = dots[i];
                let x1 = d.x * Math.cos(angleY) - d.z * Math.sin(angleY);
                let z1 = d.z * Math.cos(angleY) + d.x * Math.sin(angleY);
                d.x = x1;
                d.z = z1;

                let scale = 200 / (200 + d.z);
                let px = d.x * scale + cx;
                let py = d.y * scale + cy;

                if (d.z > -50) {
                    ctx.beginPath();
                    ctx.arc(px, py, 1.5 * scale, 0, Math.PI * 2);
                    ctx.fillStyle = d.z > 20 ? 'rgba(56, 189, 248, 0.8)' : 'rgba(168, 85, 247, 0.45)';
                    ctx.fill();
                }
            }
            requestAnimationFrame(render);
        }
        render();
    </script>
""", height=165)

if not assets_loaded:
    st.error("⚠️ **Model files missing!** Please check that `models/skillmax_model.pkl` and `models/tfidf_vectorizer.pkl` exist.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. Interactive Workspace Tabs
# -----------------------------------------------------------------------------
tab_input, tab_analytics, tab_skills, tab_debug = st.tabs([
    "🔍 Job Input & Analysis", 
    "📊 Classification Analytics", 
    "🛠️ Skill Breakdown", 
    "⚙ NLP Debug"
])

# -----------------------------------------------------------------------------
# TAB 1: Job Input & Analysis (Matching Preview Layout)
# -----------------------------------------------------------------------------
with tab_input:
    col_main, col_stats = st.columns([2.3, 1])

    with col_main:
        st.markdown("#### 📝 **Paste Job Posting Text**")
        
        btn_col1, _, _ = st.columns([1, 1, 3])
        with btn_col1:
            st.button(
                "🔄 Reset", 
                use_container_width=True, 
                type="secondary", 
                on_click=reset_workspace_callback
            )
        
        job_description = st.text_area(
            label="Job Posting Text",
            key="input_text",
            placeholder="Paste full IT job posting text here (including duties, qualifications, and stack requirements)...",
            height=280,
            label_visibility="collapsed"
        )

        analyze_btn = st.button("🚀 Analyze Job Posting", type="primary", use_container_width=True)

    with col_stats:
        st.markdown("#### 📊 **TEXT METRICS**")
        
        word_count = len(job_description.split()) if job_description.strip() else 0
        char_count = len(job_description) if job_description.strip() else 0

        st.markdown(f"""
            <div class="glass-card">
                <div class="stat-box" style="margin-bottom: 12px;">
                    <div class="stat-number">{word_count}</div>
                    <div class="stat-label">WORD COUNT</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{char_count}</div>
                    <div class="stat-label">CHARACTERS</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-card" style="padding: 18px;">
                <h5 style="color:#C084FC; font-size:0.8rem; font-weight:700; margin-bottom:8px;">🎯 SYSTEM CONTEXT</h5>
                <div style="font-size:0.75rem; color:#94A3B8; display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span>Institution:</span> <strong style="color:#F8FAFC;">UNO - Recoletos</strong>
                </div>
                <div style="font-size:0.75rem; color:#94A3B8; display:flex; justify-content:space-between;">
                    <span>Classifier:</span> <strong style="color:#38BDF8;">Linear SVC</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Result Processing Logic
# -----------------------------------------------------------------------------
if analyze_btn:
    if not job_description.strip():
        st.warning("⚠️ Please paste a valid job description before clicking analyze.")
        st.session_state["analyzed"] = False
    else:
        st.session_state["analyzed"] = True

if st.session_state.get("analyzed") and job_description.strip():
    with st.spinner("Executing NLP pre-processing & ML inference..."):
        cleaned_text = clean_input_text(job_description)
        text_vector = vectorizer.transform([cleaned_text])
        
        predicted_role = model.predict(text_vector)[0]
        decision_scores = model.decision_function(text_vector)[0]
        classes = model.classes_
        
        probabilities = softmax(decision_scores)
        top_indices = decision_scores.argsort()[-5:][::-1]
        
        top_matches = [
            {
                "Role": classes[i],
                "Score": float(decision_scores[i]),
                "Probability": float(probabilities[i] * 100)
            } 
            for i in top_indices
        ]
        
        unique_skills = extract_matched_skills(cleaned_text, vectorizer, max_skills=25)

    with tab_input:
        st.markdown("---")
        st.markdown(f"""
            <div class="glass-card">
                <span class="prediction-badge">Primary Predicted Category</span>
                <div class="predicted-role-title">👨‍💻 {predicted_role}</div>
                <p style="color: #94A3B8; margin-bottom: 0;">
                    Top Score Index: <strong style="color: #38BDF8;">{top_matches[0]['Score']:.2f}</strong> | 
                    Identified Skill Tokens: <strong style="color: #C084FC;">{len(unique_skills)} phrases</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # TAB 2: Classification Analytics
    # -----------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("### 🎯 **Model Category Confidence Breakdown**")
        
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            df_top = pd.DataFrame(top_matches).sort_values(by="Score", ascending=True)
            
            fig = px.bar(
                df_top,
                x="Score",
                y="Role",
                orientation='h',
                text_auto='.2f',
                title="Top Candidate Category Decision Scores",
                color="Score",
                color_continuous_scale=["#6D28D9", "#8B5CF6", "#38BDF8", "#E9D5FF"]
            )
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", family="Plus Jakarta Sans"),
                xaxis=dict(showgrid=True, gridcolor="rgba(168, 85, 247, 0.15)"),
                yaxis=dict(showgrid=False),
                coloraxis_showscale=False,
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            fig.update_traces(
                marker_line_color='rgba(255,255,255,0.2)',
                marker_line_width=1,
                opacity=0.95
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 📈 **Ranked Candidate Scores**")
            display_df = pd.DataFrame(top_matches)[["Role", "Score", "Probability"]]
            display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x:.1f}%")
            display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                display_df,
                column_config={
                    "Role": "Job Category",
                    "Score": "Decision Score",
                    "Probability": "Approx Fit %"
                },
                use_container_width=True,
                hide_index=True
            )

    # -----------------------------------------------------------------------------
    # TAB 3: Extracted Technical Skills
    # -----------------------------------------------------------------------------
    with tab_skills:
        st.markdown("### 🛠️ **Identified Technical Stack Tokens**")
        
        if unique_skills:
            skills_html = "".join([f'<span class="skill-pill">⚡ {skill}</span>' for skill in unique_skills])
            st.markdown(f'<div style="margin-bottom: 25px;">{skills_html}</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🔍 **Skill Keyword Frequency Search**")
            skill_counts = [{"Skill Keyword": skill, "Occurrences": cleaned_text.split().count(skill)} for skill in unique_skills]
            df_skills = pd.DataFrame(skill_counts).sort_values(by="Occurrences", ascending=False)
            
            st.dataframe(df_skills, use_container_width=True, hide_index=True)
        else:
            st.info("No specific technical terms identified.")

        st.markdown("---")
        st.markdown("### 📥 **Download Analysis Report**")
        
        report_text = (
            f"=================================================\n"
            f"           SKILLMAX AI ANALYSIS REPORT           \n"
            f"=================================================\n\n"
            f"Primary Role Prediction: {predicted_role}\n"
            f"Top Score Index        : {top_matches[0]['Score']:.2f}\n"
            f"Word Count             : {word_count}\n"
            f"Extracted Skill Count  : {len(unique_skills)}\n\n"
            f"-------------------------------------------------\n"
            f"IDENTIFIED TECHNICAL SKILLS:\n"
            f"-------------------------------------------------\n"
            f"{', '.join(unique_skills) if unique_skills else 'None identified'}\n\n"
            f"-------------------------------------------------\n"
            f"RANKED CANDIDATE CATEGORIES:\n"
            f"-------------------------------------------------\n"
        )
        for m in top_matches:
            report_text += f"- {m['Role']:<28} | Score: {m['Score']:.2f} | Approx Fit: {m['Probability']:.1f}%\n"

        st.download_button(
            label="📥 Download Analysis (.txt)",
            data=report_text,
            file_name=f"skillmax_{predicted_role.lower().replace(' ', '_')}_analysis.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary"
        )

    # -----------------------------------------------------------------------------
    # TAB 4: NLP Preprocessing Debug Pipeline
    # -----------------------------------------------------------------------------
    with tab_debug:
        st.markdown("### ⚙️ **NLP Normalization & Debug Log**")
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("**Raw Input Text Snippet:**")
            st.info(job_description[:500] + ("..." if len(job_description) > 500 else ""))
        with d_col2:
            st.markdown("**Cleaned & Tokenized Stream:**")
            st.success(cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else ""))
        
        with st.expander("🔍 View Complete Cleaned Log"):
            st.text_area("Cleaned Text Output", cleaned_text, height=180, disabled=True)
