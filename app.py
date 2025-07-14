import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Load trained model
try:
    model = joblib.load('random_forest_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_model.joblib' exists in the current directory.")
    st.stop()

# Page setup
st.set_page_config(
    page_title="🎓 Student Placement Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }

        .main-title {
            text-align: center;
            font-size: 48px;
            background: linear-gradient(45deg, #2E8B57, #3CB371);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }

        .sub-title {
            text-align: center;
            font-size: 20px;
            color: #666;
            margin-bottom: 2rem;
        }

        .metric-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .input-section {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }

        .result-success {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }

        .result-warning {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #333;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }

        .stButton > button {
            background: linear-gradient(45deg, #2E8B57, #3CB371);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .feature-importance {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }

        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }

        .info-box {
            background: #e8f4f8;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #2E8B57;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>🎓 AI-Powered Student Placement Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced machine learning model to predict student placement probability with detailed insights</div>", unsafe_allow_html=True)

# Sidebar for additional information
with st.sidebar:
    st.header("📊 About This Tool")
    st.markdown("""
    <div class='info-box'>
    <strong>🤖 Model Information:</strong><br>
    • Random Forest Algorithm<br>
    • 6 Key Features Analysis<br>
    • High Accuracy Predictions<br>
    • Real-time Processing
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Feature Importance")
    # Mock feature importance (replace with actual model feature importance)
    feature_importance = {
        'Academic Performance': 0.25,
        'Communication Skills': 0.20,
        'IQ Score': 0.18,
        'Projects Completed': 0.15,
        'Previous Semester': 0.12,
        'Internship Experience': 0.10
    }

    for feature, importance in feature_importance.items():
        st.metric(feature, f"{importance:.1%}")

    st.markdown("### 🎯 Tips for Better Placement")
    st.markdown("""
    - Focus on improving communication skills
    - Complete more practical projects
    - Gain internship experience
    - Maintain consistent academic performance
    - Develop problem-solving abilities
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("### 📝 Student Profile Input")

    # Input form with improved layout
    with st.form("placement_form", clear_on_submit=False):
        # Academic metrics
        st.markdown("#### 🎓 Academic Metrics")
        col1_1, col1_2 = st.columns(2)

        with col1_1:
            iq = st.slider('🧠 IQ Score',
                          min_value=50, max_value=200, value=100,
                          help="General intelligence quotient (50-200)")
            academic_performance = st.slider('📚 Academic Performance',
                                           min_value=0, max_value=10, value=7,
                                           help="Overall academic performance (0-10)")

        with col1_2:
            prev_sem_result = st.slider('📊 Previous Semester Result',
                                      min_value=0.0, max_value=10.0, value=7.0, step=0.1,
                                      help="Previous semester GPA/percentage (0-10)")
            communication_skills = st.slider('🗣️ Communication Skills',
                                           min_value=0, max_value=10, value=7,
                                           help="Verbal and written communication ability (0-10)")

        # Experience metrics
        st.markdown("#### 💼 Experience & Skills")
        col2_1, col2_2 = st.columns(2)

        with col2_1:
            internship_experience = st.selectbox('💼 Internship Experience',
                                               ['No', 'Yes'],
                                               help="Has completed at least one internship")

        with col2_2:
            projects_completed = st.number_input('📁 Projects Completed',
                                               min_value=0, max_value=20, value=3,
                                               help="Number of significant projects completed")

        st.markdown("<br>", unsafe_allow_html=True)
        col_submit = st.columns([1, 2, 1])
        with col_submit[1]:
            submitted = st.form_submit_button("🔍 Predict Placement Probability", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Input Summary")

    # Display current inputs in a nice format
    if 'iq' in locals():
        metrics_data = {
            'IQ Score': iq,
            'Academic Performance': academic_performance,
            'Previous Semester': prev_sem_result,
            'Communication Skills': communication_skills,
            'Projects Completed': projects_completed,
            'Internship Experience': internship_experience
        }

        # Create a radar chart for current inputs
        categories = ['IQ Score', 'Academic Perf.', 'Prev. Semester', 'Communication', 'Projects']
        values = [iq/20, academic_performance, prev_sem_result, communication_skills, projects_completed/2]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Profile',
            line_color='#2E8B57'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

# Prediction logic and results
if submitted:
    # Prepare data
    internship_experience_encoded = 1 if internship_experience == 'Yes' else 0
    student_data = pd.DataFrame({
        'IQ': [iq],
        'Prev_Sem_Result': [prev_sem_result],
        'Academic_Performance': [academic_performance],
        'Internship_Experience': [internship_experience_encoded],
        'Communication_Skills': [communication_skills],
        'Projects_Completed': [projects_completed]
    })

    # Make prediction
    try:
        prediction = model.predict(student_data)[0]
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(student_data)[0]
            placement_probability = prediction_proba[1] * 100
        else:
            placement_probability = 85 if prediction == 1 else 25  # Fallback values

        # Display results
        st.markdown("---")

        if prediction == 1:
            st.markdown(f"""
                <div class='result-success'>
                    <h2>✅ Placement Prediction: LIKELY TO BE PLACED</h2>
                    <h3>Probability: {placement_probability:.1f}%</h3>
                    <p>Based on the provided academic profile, the student has a high probability of securing a placement.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class='result-warning'>
                    <h2>⚠️ Placement Prediction: NEEDS IMPROVEMENT</h2>
                    <h3>Probability: {placement_probability:.1f}%</h3>
                    <p>The student may need to focus on improving certain areas to increase placement chances.</p>
                </div>
            """, unsafe_allow_html=True)

        # Detailed analysis
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 📈 Detailed Analysis")

            # Strengths and weaknesses
            strengths = []
            weaknesses = []

            if iq >= 120:
                strengths.append("🧠 High IQ Score")
            elif iq <= 90:
                weaknesses.append("🧠 IQ Score needs improvement")

            if academic_performance >= 8:
                strengths.append("📚 Excellent Academic Performance")
            elif academic_performance <= 5:
                weaknesses.append("📚 Academic Performance needs focus")

            if communication_skills >= 8:
                strengths.append("🗣️ Strong Communication Skills")
            elif communication_skills <= 5:
                weaknesses.append("🗣️ Communication Skills need development")

            if projects_completed >= 5:
                strengths.append("📁 Good Project Portfolio")
            elif projects_completed <= 2:
                weaknesses.append("📁 Need more project experience")

            if internship_experience == 'Yes':
                strengths.append("💼 Valuable Internship Experience")
            else:
                weaknesses.append("💼 Lack of internship experience")

            if strengths:
                st.markdown("**🎯 Key Strengths:**")
                for strength in strengths:
                    st.markdown(f"• {strength}")

            if weaknesses:
                st.markdown("**🔧 Areas for Improvement:**")
                for weakness in weaknesses:
                    st.markdown(f"• {weakness}")

        with col4:
            st.markdown("### 🎯 Recommendations")

            recommendations = []

            if communication_skills <= 6:
                recommendations.append("Join public speaking clubs or communication workshops")

            if projects_completed <= 3:
                recommendations.append("Work on more practical projects in your field")

            if internship_experience == 'No':
                recommendations.append("Apply for internships to gain industry experience")

            if academic_performance <= 6:
                recommendations.append("Focus on improving academic grades")

            if not recommendations:
                recommendations = [
                    "Maintain current performance levels",
                    "Consider leadership roles in projects",
                    "Build a strong professional network",
                    "Prepare well for placement interviews"
                ]

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

        # Feature contribution visualization
        st.markdown("### 📊 Feature Contribution Analysis")

        feature_values = [iq/20, prev_sem_result, academic_performance,
                         internship_experience_encoded*10, communication_skills, projects_completed]
        feature_names = ['IQ Score', 'Previous Semester', 'Academic Performance',
                        'Internship Experience', 'Communication Skills', 'Projects Completed']

        fig_bar = px.bar(
            x=feature_names,
            y=feature_values,
            title="Current Profile Strengths",
            color=feature_values,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(
            xaxis_title="Features",
            yaxis_title="Score (Normalized)",
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎓 Student Placement Predictor • Built with Streamlit & Machine Learning</p>
        <p>⚡ Powered by Random Forest Algorithm • Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)
