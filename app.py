import streamlit as st
import re
import nltk
import pandas as pd
import os
from nltk.corpus import stopwords

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkillMax",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished Theme Styling
st.markdown("""
    <style>
    /* Main Title & Subtitle */
    .main-title {
        font-size: 2.6rem !important;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Header Branding Area */
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 10px;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 15px;
    }
    .predicted-role {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 5px;
    }
    .badge {
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Custom Skill Tags */
    .skill-tag {
        display: inline-block;
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #0284C7;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 18px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Sidebar Section Headers */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        padding-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Asset Loading & Preprocessing Functions
# -----------------------------------------------------------------------------
nltk.download('stopwords', quiet=True)
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

def extract_matched_skills(cleaned_text, vectorizer, max_skills=20):
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

SAMPLE_DESCRIPTIONS = {
    "-- Choose a Sample --": "",
    "Data Scientist": "We are seeking a Data Scientist with strong Python, SQL, and Machine Learning experience. Must be familiar with pandas, scikit-learn, neural networks, and data visualization.",
    "Big Data Engineer": "Looking for a DevOps Engineer experienced in AWS, Docker, Kubernetes, Terraform, CI/CD pipelines, and Linux administration.",
    "Full Stack Developer": "Hiring a Full Stack Web Developer skilled in JavaScript, React, Node.js, HTML5, CSS3, REST APIs, and MongoDB database management."
}

LOGO_PATH = "assets/unor_logo.png"

# -----------------------------------------------------------------------------
# 3. Sidebar Setup
# -----------------------------------------------------------------------------
with st.sidebar:
    # Display UNO-R Logo in Sidebar if file exists
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)
    else:
        st.caption("📍 *Place UNO-R logo image in assets/unor_logo.png*")

    st.markdown("<p class='sidebar-header'>⚡ SkillMax Assistant</p>", unsafe_allow_html=True)
    st.markdown(
        "**SkillMax** uses Natural Language Processing (NLP) and Machine Learning "
        "to automatically categorize unstructured IT job listings and discover target tech stacks."
    )
    st.divider()
    
    st.markdown("### 🧪 Quick Test Samples")
    selected_sample = st.selectbox("Load Sample Job Description:", list(SAMPLE_DESCRIPTIONS.keys()))
    
    st.divider()
    st.markdown("### 📊 Project Metadata")
    st.write("**Institution:** UNO - Recoletos")
    st.write("**Department:** College of IT")
    st.write("**Model:** Linear Support Vector Classifier")
    st.write("**Vectorizer:** TF-IDF (Unigrams & Bigrams)")
    st.write("**Dataset Scope:** 10,000 Posts across 25 Roles")
    st.divider()
    
    st.caption("Developed by Group **DATA-MAX**")

# -----------------------------------------------------------------------------
# 4. Main User Interface Header
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([0.1, 10])

with header_col2:
    st.markdown("<h1 class='main-title'>⚡ SkillMax Job Classifier & Skill Extractor Assistant</h1>", unsafe_allow_html=True)

if not assets_loaded:
    st.error("⚠️ **Model files missing!** Please run your training notebook (`01_eda_and_training.ipynb`) to save the model and vectorizer in the `models/` directory.")
    st.stop()

default_text = SAMPLE_DESCRIPTIONS[selected_sample] if selected_sample != "-- Choose a Sample --" else ""

# Layout: 2 Columns for Input and Quick Stats
col1, col2 = st.columns([2.2, 1])

with col1:
    job_description = st.text_area(
        "📋 Paste Job Description Text",
        value=default_text,
        placeholder="Paste full IT job posting text here (including duties, qualifications, and requirements)...",
        height=260
    )
    analyze_btn = st.button("🚀 Analyze Job Description", type="primary", use_container_width=True)

with col2:
    st.markdown("### 💡 Quick Tip")
    st.info(
        "For the most accurate classification results, make sure to include the full text block containing key duties, technology stack details, and required qualifications."
    )
    st.metric(label="Supported Domains", value="25 IT Categories")
    st.metric(label="Target Model", value="LinearSVC + TF-IDF")

# -----------------------------------------------------------------------------
# 5. Result Processing & Presentation
# -----------------------------------------------------------------------------
if analyze_btn:
    if not job_description.strip():
        st.warning("⚠️ Please paste a valid job description before clicking analyze.")
    else:
        with st.spinner("Processing text and executing NLP model..."):
            cleaned_text = clean_input_text(job_description)
            
            text_vector = vectorizer.transform([cleaned_text])
            predicted_role = model.predict(text_vector)[0]
            
            decision_scores = model.decision_function(text_vector)[0]
            classes = model.classes_
            top_3_idx = decision_scores.argsort()[-3:][::-1]
            top_matches = [(classes[i], decision_scores[i]) for i in top_3_idx]
            
            unique_skills = extract_matched_skills(cleaned_text, vectorizer, max_skills=20)

        st.divider()
        st.markdown("## 🎯 Classification Results")
        
        res_col1, res_col2 = st.columns([1.6, 1])
        
        with res_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <span class="badge">Primary Predicted Category</span>
                    <div class="predicted-role">👨‍💻 {predicted_role}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with res_col2:
            st.metric(label="Word Count (Raw)", value=f"{len(job_description.split())} words")
            st.metric(label="Extracted Skill Tokens", value=f"{len(unique_skills)} key phrases")

        st.markdown("### 📊 Top Candidate Role Match Scores")
        top_df = pd.DataFrame(top_matches, columns=["Role Category", "Decision Confidence Score"])
        top_df["Decision Confidence Score"] = top_df["Decision Confidence Score"].apply(lambda x: f"{x:.2f}")
        st.table(top_df)

        st.markdown("### 🛠️ Identified Technical Skill Keywords")
        if unique_skills:
            tags_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in unique_skills])
            st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.write("No major domain-specific technical terms identified.")

        st.divider()
        report_text = f"=========================\n" \
                      f"SKILLMAX ANALYSIS REPORT\n" \
                      f"=========================\n" \
                      f" \n" \
                      f"Primary Role Prediction: {predicted_role}\n" \
                      f"Word Count: {len(job_description.split())} words\n\n" \
                      f"Identified Technical Skills:\n" \
                      f"{', '.join(unique_skills) if unique_skills else 'None identified'}\n\n" \
                      f"Top Category Decision Scores:\n" \
                      + "\n".join([f"- {role}: {score:.2f}" for role, score in top_matches])

        st.download_button(
            label="📥 Download Analysis Report (.txt)",
            data=report_text,
            file_name=f"skillmax_analysis_{predicted_role.lower().replace(' ', '_')}.txt",
            mime="text/plain"
        )

        with st.expander("🔍 View Preprocessed Text Debug Log"):
            st.text_area("Cleaned & Tokenized Input", cleaned_text, height=120)
