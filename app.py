import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import os
import numpy as np
import plotly.express as px
from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# 1. Page Configuration & Synaptix Violet Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkillMax AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Violet & Purple UI Theme Styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background Styling */
    .stApp {
        background-color: #0B0713 !important;
        color: #E2E8F0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #120A1F !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
    }
    
    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(20, 12, 34, 0.95) 0%, rgba(32, 17, 56, 0.85) 100%);
        border: 1px solid rgba(168, 85, 247, 0.2);
        border-radius: 20px;
        padding: 26px 32px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(139, 92, 246, 0.1);
        margin-bottom: 25px;
    }

    .main-title {
        font-size: 2.2rem !important;
        font-weight: 800;
        background: linear-gradient(120deg, #FFFFFF, #C084FC, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .sub-title {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 0px;
        font-weight: 500;
    }

    /* Synaptix Cards */
    .glass-card {
        background: rgba(20, 12, 34, 0.7);
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-radius: 18px;
        padding: 22px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        transition: all 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.15);
    }

    /* Badges & Pills */
    .prediction-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
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
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 8px;
    }

    .skill-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(139, 92, 246, 0.15);
        color: #D8B4FE;
        border: 1px solid rgba(168, 85, 247, 0.35);
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .skill-pill:hover {
        background: rgba(139, 92, 246, 0.3);
        border-color: #C084FC;
        transform: translateY(-2px);
    }

    /* Streamlit Interactive Primary Buttons (Pill Shaped) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.5) !important;
        transform: translateY(-1px);
    }

    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #C084FC !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(20, 12, 34, 0.8);
        padding: 6px;
        border-radius: 30px;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 20px;
        padding: 8px 20px;
        color: #94A3B8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }

    /* Stat Box */
    .stat-box {
        background: rgba(12, 7, 19, 0.6);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(168, 85, 247, 0.15);
        text-align: center;
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    .stat-label {
        font-size: 0.78rem;
        color: #A855F7;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
    "DevOps / Cloud Engineer": "Looking for a Cloud DevOps Engineer experienced in AWS, Docker, Kubernetes, Terraform, Jenkins, CI/CD automated pipelines, bash scripting, and Linux system administration. Experience with Ansible and Prometheus is a plus.",
    "Full Stack Web Developer": "Hiring a Full Stack Web Developer skilled in JavaScript, TypeScript, React.js, Node.js, Express, HTML5, CSS3, RESTful APIs, GraphQL, and MongoDB database design. AWS experience preferred.",
    "Cybersecurity Analyst": "Seeking a Cybersecurity Analyst to monitor security posture, conduct vulnerability assessments, analyze malware, configure firewalls, manage SIEM tools, and enforce compliance frameworks like ISO27001."
}

LOGO_PATH = "assets/unor_logo.png"

# Initialize Session State
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False

# Callback when selecting a dropdown sample
def on_sample_select():
    selected = st.session_state["sample_choice"]
    if selected != "-- Select an Interactive Sample --":
        st.session_state["input_text"] = SAMPLE_DESCRIPTIONS[selected]
    # Ensure previous analysis results do NOT automatically display
    st.session_state["analyzed"] = False

# -----------------------------------------------------------------------------
# 3. Sidebar UI Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("<h3 style='color:#C084FC;'>🏛️ UNO - Recoletos</h3>", unsafe_allow_html=True)
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
    st.markdown("### 📋 Project Details")
    st.markdown("""
    - **Institution:** UNO - Recoletos
    - **Department:** College of IT
    - **Classifier:** Linear SVC
    - **Vectorization:** TF-IDF (N-Grams 1-2)
    - **Scope:** 25 IT Domains
    """)
    st.markdown("---")
    st.caption("Developed by **Group DATA-MAX**")

# -----------------------------------------------------------------------------
# 4. Hero Section Header
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div class="main-title">⚡ SkillMax AI Assistant</div>
        <div class="sub-title">Automated Job Category Classification & Skill Extraction Dashboard</div>
    </div>
""", unsafe_allow_html=True)

if not assets_loaded:
    st.error("⚠️ **Model files missing!** Please check that `models/skillmax_model.pkl` and `models/tfidf_vectorizer.pkl` exist.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Interactive Workspace Tabs
# -----------------------------------------------------------------------------
tab_input, tab_analytics, tab_skills, tab_debug = st.tabs([
    "🔍 Job Input & Analysis", 
    "📊 Classification Analytics", 
    "🛠️ Skill Breakdown & Export", 
    "⚙️ NLP Debug Pipeline"
])

# -----------------------------------------------------------------------------
# TAB 1: Job Description Input & Reset Control
# -----------------------------------------------------------------------------
with tab_input:
    col_main, col_stats = st.columns([2.3, 1])

    # Callback function to reset input text, stats, and previous results
    def reset_workspace_callback():
        st.session_state["input_text"] = ""
        st.session_state["analyzed"] = False
        st.session_state["sample_choice"] = "-- Select an Interactive Sample --"

    with col_main:
        st.markdown("#### 📝 **Input Job Description**")
        
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

        analyze_btn = st.button("🚀 Analyze Job Description", type="primary", use_container_width=True)

    with col_stats:
        st.markdown("#### 📊 **Quick Text Stats**")
        
        # Calculates word/char count for current text input
        word_count = len(job_description.split()) if job_description.strip() else 0
        char_count = len(job_description) if job_description.strip() else 0

        st.markdown(f"""
            <div class="glass-card">
                <div class="stat-box" style="margin-bottom: 12px;">
                    <div class="stat-number">{word_count}</div>
                    <div class="stat-label">Word Count</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{char_count}</div>
                    <div class="stat-label">Character Count</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Result Processing Logic (Only triggers if Analyze button is clicked)
# -----------------------------------------------------------------------------
if analyze_btn:
    if not job_description.strip():
        st.warning("⚠️ Please paste a valid job description before clicking analyze.")
        st.session_state["analyzed"] = False
    else:
        st.session_state["analyzed"] = True

if st.session_state.get("analyzed") and job_description.strip():
    with st.spinner("Executing NLP pre-processing & ML model..."):
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
                    Top Score Index: <strong style="color: #C084FC;">{top_matches[0]['Score']:.2f}</strong> | 
                    Identified Skill Tokens: <strong style="color: #C084FC;">{len(unique_skills)} phrases</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------------
    # TAB 2: Classification Analytics
    # -----------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("### 🎯 **Top Candidate Category Matches**")
        
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            df_top = pd.DataFrame(top_matches).sort_values(by="Score", ascending=True)
            
            fig = px.bar(
                df_top,
                x="Score",
                y="Role",
                orientation='h',
                text_auto='.2f',
                title="Top Category Confidence Scores",
                color="Score",
                color_continuous_scale=["#6D28D9", "#8B5CF6", "#C084FC", "#E9D5FF"]
            )
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", family="Plus Jakarta Sans"),
                xaxis=dict(showgrid=True, gridcolor="rgba(168, 85, 247, 0.1)"),
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
            st.markdown("#### 📈 **Model Score Table**")
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
        st.markdown("### 🛠️ **Identified Technical Skill Keywords**")
        
        if unique_skills:
            skills_html = "".join([f'<span class="skill-pill">⚡ {skill}</span>' for skill in unique_skills])
            st.markdown(f'<div style="margin-bottom: 25px;">{skills_html}</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🔍 **Skill Term Occurrences**")
            skill_counts = [{"Skill Keyword": skill, "Occurrences": cleaned_text.split().count(skill)} for skill in unique_skills]
            df_skills = pd.DataFrame(skill_counts).sort_values(by="Occurrences", ascending=False)
            
            st.dataframe(df_skills, use_container_width=True, hide_index=True)
        else:
            st.info("No specific technical terms identified.")

        st.markdown("---")
        st.markdown("### 📥 **Download Analysis Report**")
        
        report_text = (
            f"=================================================\n"
            f"           SKILLMAX ANALYSIS REPORT              \n"
            f"=================================================\n\n"
            f"Primary Role Prediction: {predicted_role}\n"
            f"Decision Score         : {top_matches[0]['Score']:.2f}\n"
            f"Word Count             : {word_count}\n"
            f"Extracted Skill Count  : {len(unique_skills)}\n\n"
            f"-------------------------------------------------\n"
            f"IDENTIFIED SKILLS:\n"
            f"-------------------------------------------------\n"
            f"{', '.join(unique_skills) if unique_skills else 'None identified'}\n\n"
            f"-------------------------------------------------\n"
            f"TOP CATEGORY SCORES:\n"
            f"-------------------------------------------------\n"
        )
        for m in top_matches:
            report_text += f"- {m['Role']:<25} | Score: {m['Score']:.2f} | Approx Fit: {m['Probability']:.1f}%\n"

        st.download_button(
            label="📥 Download Analysis Report (.txt)",
            data=report_text,
            file_name=f"skillmax_{predicted_role.lower().replace(' ', '_')}_report.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary"
        )

    # -----------------------------------------------------------------------------
    # TAB 4: Debug Log
    # -----------------------------------------------------------------------------
    with tab_debug:
        st.markdown("### ⚙️ **NLP Token Pipeline**")
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("**Raw Text Sample:**")
            st.info(job_description[:500] + ("..." if len(job_description) > 500 else ""))
        with d_col2:
            st.markdown("**Preprocessed Tokens:**")
            st.success(cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else ""))
        
        with st.expander("🔍 View Complete Cleaned Log"):
            st.text_area("Cleaned Text Output", cleaned_text, height=180, disabled=True)
