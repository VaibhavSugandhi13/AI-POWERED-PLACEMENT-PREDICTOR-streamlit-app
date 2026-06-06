import pandas as pd
import streamlit as st
import pickle
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎯",
    layout="centered"
)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div style="text-align:center; padding:20px 10px;">
    <h1 style="margin-bottom:5px;
        background: linear-gradient(90deg,#4F46E5,#06B6D4);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;">
        🎯 AI Placement Predictor
    </h1>
    <p style="color:gray; font-size:16px;">
        Simple, fast & AI-powered placement prediction
    </p>
    <p style="color:#aaa;">Welcome Vaibhav Sugandhi 👋</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# LOAD MODEL
# =========================


# =========================
# LOAD MODEL
# =========================

# 1. Get the directory where app.py is currently running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Setup fallbacks to check for the models directory location
path_option_1 = os.path.join(BASE_DIR, '..', 'models', 'placement_model.pkl') # Local structure
path_option_2 = os.path.join(BASE_DIR, 'models', 'placement_model.pkl')      # Cloud deployment fallback
path_option_3 = os.path.join('models', 'placement_model.pkl')                # Root execution fallback

# 3. Choose whichever path is valid on the active server environment
if os.path.exists(path_option_1):
    model_path = path_option_1
elif os.path.exists(path_option_2):
    model_path = path_option_2
else:
    model_path = path_option_3

# 4. Load the model safely
with open(model_path, "rb") as f:
    model = pickle.load(f)
# =========================
# UI CARD STYLE
# =========================
st.markdown("""
<style>
.card {
    background: #0f172a;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚀 Build Your AI Career Profile")

# =========================
# CARD 1 - ACADEMICS
# =========================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Academics")

    col1, col2, col3 = st.columns(3)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 6.5)
    with col2:
        ssc = st.slider("SSC Marks", 0.0, 100.0, 70.0)
    with col3:
        hsc = st.slider("HSC Marks", 0.0, 100.0, 70.0)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CARD 2 - SKILLS
# =========================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💼 Skills")

    col1, col2, col3 = st.columns(3)

    with col1:
        aptitude = st.slider("Aptitude Score", 0.0, 100.0, 60.0)
    with col2:
        softskills = st.slider("Soft Skills", 0.0, 10.0, 5.0)
    with col3:
        internships = st.select_slider("Internships", [0,1,2,3,4,5], value=0)

    projects = st.select_slider("Projects", [0,1,2,3,4,5,6,7,8], value=1)
    workshops = st.select_slider("Certifications", [0,1,2,3,4,5,6,7], value=0)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CARD 3 - ACTIVITIES
# =========================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Activities")

    col1, col2 = st.columns(2)

    with col1:
        extra = st.radio("Extracurricular Activities", ["No", "Yes"], horizontal=True)

    with col2:
        training = st.radio("Placement Training", ["No", "Yes"], horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)

# convert categorical
extra = 1 if extra == "Yes" else 0
training = 1 if training == "Yes" else 0

st.divider()

# =========================
# BUTTON
# =========================
st.markdown("### 🎯 Generate Your AI Placement Report")

predict = st.button("🚀 Predict Placement", use_container_width=True)

# =========================
# PREDICTION
# =========================
if predict:

    data = pd.DataFrame([[cgpa, internships, projects, workshops,
                          aptitude, softskills, extra, training, ssc, hsc]],
                        columns=[
                            'CGPA',
                            'Internships',
                            'Projects',
                            'Workshops/Certifications',
                            'AptitudeTestScore',
                            'SoftSkillsRating',
                            'ExtracurricularActivities',
                            'PlacementTraining',
                            'SSC_Marks',
                            'HSC_Marks'
                        ])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1] * 100

    st.divider()

    # =========================
    # RESULT CARD
    # =========================
    color = "#22c55e" if prediction[0] == 1 else "#ef4444"
    status = "🎉 Placement Likely" if prediction[0] == 1 else "⚠️ Placement Unlikely"

    st.markdown(f"""
        <div style="
            text-align:center;
            padding:20px;
            border-radius:15px;
            background-color:{color}15;
            border:2px solid {color};
        ">
            <h3 style="margin-bottom:5px;">{status}</h3>
            <h2>{round(probability,2)}%</h2>
        </div>
    """, unsafe_allow_html=True)

    st.progress(int(probability))

    # =========================
    # GUIDANCE
    # =========================
    st.markdown("### 🧠 Career Guidance")

    if probability < 50:
        st.error("Focus on DSA, Projects & Internships")
    else:
        st.success("Good chances! Focus on interview preparation")

    # =========================
    # IMPROVEMENT TIPS
    # =========================
    st.markdown("### 💡 Improvement Tips")

    if aptitude < 70:
        st.info("Improve Aptitude Skills")

    if softskills < 7:
        st.info("Improve Communication Skills")

    if projects < 2:
        st.info("Build More Projects")

    if internships == 0:
        st.info("Do at least 1 Internship")
    else:
        st.info("Keep improving DSA + Interview Skills")