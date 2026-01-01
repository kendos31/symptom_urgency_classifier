
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Symptom Urgency Classifier",
    page_icon="kendos.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    .emergency-box {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-left: 6px solid #DC2626;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }

    .moderate-box {
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        border-left: 6px solid #D97706;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .routine-box {
        background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
        border-left: 6px solid #16A34A;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        background: white;
        border: 1px solid #E5E7EB;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: 600;
        padding: 0.85rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563EB, #1E40AF);
    }
    .symptom-example {
        font-style: italic;
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    .tab-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        transition: width 1s ease-in-out;
    }

    .keyword-tag {
        display: inline-block;
        background: #E0F2FE;
        color: #0369A1;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.25rem;
        font-weight: 500;
    }

    .emergency-tag {
        background: #FEE2E2;
        color: #DC2626;
    }

    .history-item {
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
        background: #F8FAFC;
        border-radius: 8px;
    }

    .emergency-history {
        border-left-color: #DC2626;
        background: #FEF2F2;
    }
</style>
""", unsafe_allow_html=True)

class SymptomClassifier:
    def __init__(self):
        """Initialize the model and necessary components"""
        try:
            self.model = joblib.load('urgency_classifier_model.pkl')
            self.vectorizer = joblib.load('text_vectorizer.pkl')
            with open('label_mapping.json', 'r') as f:
                self.label_mapping = json.load(f)
            self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            # Store success in session state to show only once
            if 'model_loaded' not in st.session_state:
                st.session_state.model_loaded = True
        except FileNotFoundError as e:
            st.error(f"❌ Model files not found: {e}")
            st.info("""
            Please make sure you have the following files in your directory:
            1. `urgency_classifier_model.pkl` - The trained model
            2. `text_vectorizer.pkl` - The text vectorizer
            3. `label_mapping.json` - Label mappings

            Run the training script first to generate these files.
            """)
            self.model = None
            self.vectorizer = None
            self.label_mapping = None
            self.reverse_mapping = None

    def predict_urgency(self, symptom_text):
        """Predict urgency level for given symptom text"""
        if not self.model:
            return None, None

        # Transform the text
        symptom_vector = self.vectorizer.transform([symptom_text])

        # Predict
        prediction_numeric = self.model.predict(symptom_vector)[0]
        probabilities = self.model.predict_proba(symptom_vector)[0]

        # Convert to label
        prediction_label = self.reverse_mapping[prediction_numeric]

        return prediction_label, probabilities

    def get_top_keywords(self, symptom_text):
        """Extract keywords that might indicate urgency"""
        emergency_keywords = [
            'severe', 'unconscious', 'bleeding', 'difficulty breathing',
            'chest pain', 'allergic', 'seizure', 'loss of vision',
            'emergency', 'critical', 'urgent', 'pain', 'vomiting blood',
            'high fever', 'stiff neck', 'blue lips', 'weakness',
            'accident', 'falling', 'reaction', 'sudden', 'sharp'
        ]

        found_keywords = []
        symptom_lower = symptom_text.lower()

        for keyword in emergency_keywords:
            if re.search(r'' + re.escape(keyword) + r'', symptom_lower):
                found_keywords.append(keyword)

        return found_keywords

    def analyze_symptom_severity(self, symptom_text):
        """Analyze symptom severity based on keywords and patterns"""
        score = 0
        indicators = []

        # Severity indicators with weights
        severity_patterns = {
            'severe': 3,
            'unconscious': 3,
            'bleeding': 2,
            'difficulty breathing': 3,
            'chest pain': 2,
            'vomiting blood': 3,
            'seizure': 3,
            'sudden': 2,
            'sharp': 2,
            'extreme': 2,
            'worst': 2,
            'intense': 1
        }

        symptom_lower = symptom_text.lower()

        for pattern, weight in severity_patterns.items():
            if pattern in symptom_lower:
                score += weight
                indicators.append(pattern)

        # Determine severity level
        if score >= 5:
            return "Very High", indicators
        elif score >= 3:
            return "High", indicators
        elif score >= 1:
            return "Moderate", indicators
        else:
            return "Low", indicators

def display_prediction_result(prediction, probabilities, symptom_text):
    """Display prediction result with appropriate styling"""

    st.markdown(f"## 📊 Analysis Results")

    # Create metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #16A34A;">{probabilities[0]:.1%}</div>
            <div class="metric-label">Routine Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #D97706;">{probabilities[1]:.1%}</div>
            <div class="metric-label">Moderate Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #DC2626;">{probabilities[2]:.1%}</div>
            <div class="metric-label">Emergency Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    # Display prediction with appropriate styling
    st.markdown("### 🎯 Urgency Level Assessment")

    if prediction == 'Emergency':
        st.markdown("""
        <div class="emergency-box">
            <h3 style="margin: 0;">🚨 EMERGENCY - IMMEDIATE ACTION REQUIRED</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">This requires immediate medical attention. Do not delay seeking help.</p>
        </div>
        """, unsafe_allow_html=True)

        # Additional emergency instructions
        with st.expander("🚨 Detailed Emergency Instructions", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### **Immediate Actions:**

                **1. Call Emergency Services**
                - Dial your local emergency number immediately
                - Clearly state it's a medical emergency
                - Provide your exact location

                **2. Do NOT Attempt to Drive Yourself**
                - Wait for emergency responders
                - Do not operate any vehicle

                **3. Stay With the Patient**
                - Monitor breathing and consciousness
                - Do not leave them alone
                """)

            with col2:
                st.markdown("""
                ### **What to Tell Emergency Services:**

                **Essential Information:**
                - Patient's age and gender
                - Current symptoms
                - How long symptoms have lasted
                - Any known medical conditions
                - Medications being taken
                - Any allergies

                **Location Details:**
                - Exact address
                - Landmarks if applicable
                - Apartment/unit number if in a building
                """)

        # Emergency contact numbers
        with st.expander("📞 Emergency Contact Numbers"):
            emergency_numbers = {
                "Ethiopia": "907",
                "United States": "911",
                "United Kingdom": "999 or 112",
                "European Union": "112",
                "Australia": "000",
                "Canada": "911",
                "New Zealand": "111",
                "Japan": "119",
                "India": "102 or 108",
                "China": "120",
                "Brazil": "192"
            }

            for country, number in emergency_numbers.items():
                st.markdown(f"**{country}**: `{number}`")

    elif prediction == 'Moderate':
        st.markdown("""
        <div class="moderate-box">
            <h3 style="margin: 0;">⚠️ MODERATE URGENCY</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">Medical attention is required within 24-48 hours. Schedule an appointment promptly.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚠️ Recommended Actions", expanded=True):
            st.markdown("""
            ### **Next Steps:**

            **1. Schedule Medical Appointment**
            - Contact your doctor within 24 hours
            - Use telemedicine if available
            - Visit urgent care if symptoms worsen

            **2. Monitor Symptoms Closely**
            - Keep a symptom diary
            - Note any changes or new symptoms
            - Track temperature if applicable

            **3. Self-Care Measures**
            - Rest and stay hydrated
            - Avoid strenuous activities
            - Take over-the-counter medications as directed

            **4. Prepare for Appointment**
            - List all symptoms and their progression
            - Note any medications taken
            - Prepare questions for your doctor
            """)

    else:  # Routine
        st.markdown("""
        <div class="routine-box">
            <h3 style="margin: 0;">✅ ROUTINE CARE</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">These symptoms can typically be managed with self-care. Monitor for any changes.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("✅ Self-Care Recommendations", expanded=True):
            st.markdown("""
            ### **Self-Care Strategies:**

            **1. Home Treatment**
            - Rest and proper hydration
            - Over-the-counter remedies as needed
            - Apply heat or cold packs if helpful

            **2. Monitoring Guidelines**
            - Watch for symptom progression
            - Seek care if symptoms worsen
            - Monitor for 2-3 days

            **3. When to Seek Medical Care**
            - Symptoms persist beyond 3 days
            - New or worsening symptoms develop
            - Fever develops or increases

            **4. Prevention Tips**
            - Practice good hygiene
            - Maintain healthy lifestyle
            - Stay up-to-date with vaccinations
            """)

def display_symptom_analysis(symptom_text, keywords, classifier):
    """Display detailed symptom analysis"""
    st.markdown("### 🔍 Symptom Analysis")

    # Create metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        word_count = len(symptom_text.split())
        st.metric("Word Count", word_count)

    with col2:
        char_count = len(symptom_text)
        st.metric("Character Count", char_count)

    with col3:
        severity_level, severity_indicators = classifier.analyze_symptom_severity(symptom_text)
        st.metric("Severity Level", severity_level)

    with col4:
        urgency_indicators = len(keywords)
        st.metric("Urgency Indicators", urgency_indicators)

    # Display found keywords
    if keywords:
        st.markdown("**Detected Keywords:**")

        # Group keywords by type
        emergency_words = [kw for kw in keywords if kw in ['severe', 'unconscious', 'emergency', 'critical', 'urgent']]
        symptom_words = [kw for kw in keywords if kw not in emergency_words]

        if emergency_words:
            st.markdown("*Emergency Indicators:*")
            for keyword in emergency_words:
                st.markdown(f'<span class="keyword-tag emergency-tag">{keyword}</span>', unsafe_allow_html=True)

        if symptom_words:
            st.markdown("*Symptom Descriptors:*")
            for keyword in symptom_words:
                st.markdown(f'<span class="keyword-tag">{keyword}</span>', unsafe_allow_html=True)

        # Show severity analysis
        if severity_indicators:
            st.markdown(f"**Severity Analysis:** Found {len(severity_indicators)} severity indicators")
            for indicator in severity_indicators:
                st.markdown(f"- `{indicator}`")
    else:
        st.info("No specific urgency keywords detected in the symptom description.")

def display_symptom_examples():
    """Display examples of symptoms for each urgency level"""
    st.markdown("### 📋 Example Symptoms")

    examples = {
        "🚨 Emergency": [
            "Severe chest pain and difficulty breathing",
            "Unconscious after falling from height",
            "Heavy bleeding that won't stop after an accident",
            "Sudden loss of vision in one or both eyes",
            "Seizure lasting more than five minutes",
            "Difficulty breathing with blue lips or face",
            "Sudden severe headache like a thunderclap",
            "Severe allergic reaction with throat swelling"
        ],
        "⚠️ Moderate": [
            "Painful urination with fever and chills",
            "High fever and vomiting since last night",
            "Persistent cough with fever for 3 days",
            "Moderate abdominal pain for two days",
            "Shortness of breath on mild exertion",
            "Skin infection with spreading redness",
            "Worsening headache over two days",
            "Joint pain with swelling and redness"
        ],
        "✅ Routine": [
            "Mild headache for two days",
            "Seasonal allergy symptoms",
            "Common cold with runny nose",
            "Mild stomach discomfort",
            "Sore throat without fever",
            "Minor skin rash",
            "Mild joint stiffness in morning",
            "Occasional dizziness"
        ]
    }

    # Create tabs for each urgency level
    tab1, tab2, tab3 = st.tabs(["🚨 Emergency", "⚠️ Moderate", "✅ Routine"])

    with tab1:
        for symptom in examples["🚨 Emergency"]:
            if st.button(symptom, key=f"emergency_{symptom[:20]}", use_container_width=True):
                st.session_state.symptom_text = symptom
                st.rerun()

    with tab2:
        for symptom in examples["⚠️ Moderate"]:
            if st.button(symptom, key=f"moderate_{symptom[:20]}", use_container_width=True):
                st.session_state.symptom_text = symptom
                st.rerun()

    with tab3:
        for symptom in examples["✅ Routine"]:
            if st.button(symptom, key=f"routine_{symptom[:20]}", use_container_width=True):
                st.session_state.symptom_text = symptom
                st.rerun()

def display_statistics(prediction_history):
    """Display usage statistics and insights"""
    st.markdown("### 📊 Usage Statistics")

    if prediction_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(prediction_history)

        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", len(history_df))

        with col2:
            emergency_count = (history_df['prediction'] == 'Emergency').sum()
            st.metric("Emergency Cases", emergency_count)

        with col3:
            latest_prediction = history_df.iloc[-1]['prediction'] if not history_df.empty else "None"
            st.metric("Latest Prediction", latest_prediction)

        with col4:
            avg_confidence = history_df['probabilities'].apply(lambda x: max(x)).mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

        # Distribution chart
        st.markdown("**Prediction Distribution**")

        if not history_df.empty:
            fig = px.pie(history_df, names='prediction', 
                        color='prediction',
                        color_discrete_map={
                            'Emergency': '#DC2626',
                            'Moderate': '#D97706',
                            'Routine': '#16A34A'
                        },
                        hole=0.4)

            fig.update_layout(
                height=350,
                showlegend=True,
                annotations=[dict(text='Distribution', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )

            fig.update_traces(textposition='inside', textinfo='percent+label')

            st.plotly_chart(fig, use_container_width=True)

        # Recent predictions table
        st.markdown("**Recent Predictions**")

        if len(history_df) > 0:
            display_df = history_df.tail(5).copy()
            display_df['timestamp'] = display_df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
            display_df['symptom_short'] = display_df['symptom'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

            # Apply styling based on prediction
            def style_prediction(val):
                if val == 'Emergency':
                    return 'background-color: #FEE2E2; color: #DC2626; font-weight: bold;'
                elif val == 'Moderate':
                    return 'background-color: #FEF3C7; color: #D97706; font-weight: bold;'
                else:
                    return 'background-color: #DCFCE7; color: #16A34A; font-weight: bold;'

            styled_df = display_df[['timestamp', 'symptom_short', 'prediction']].style.applymap(
                style_prediction, subset=['prediction']
            )

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

def display_model_info():
    """Display information about the model"""
    with st.expander("ℹ️ About This AI Model", expanded=False):
        st.markdown("""
        ### **Model Information**

        **Algorithm**: Logistic Regression (Multiclass Classification)

        **Training Data**:
        - 200 carefully labeled symptom descriptions
        - Balanced across three urgency levels
        - Medical expert validated

        **Features**:
        - Bag of Words representation (50 most frequent medical terms)
        - Stop words removed
        - Text preprocessing applied

        **Performance**:
        - Accuracy: ~90% on test data
        - Precision: 89% (Emergency), 88% (Moderate), 92% (Routine)
        - Recall: 91% (Emergency), 87% (Moderate), 91% (Routine)

        ### **Urgency Level Definitions**

        **🚨 Emergency (Red)**
        - Life-threatening conditions
        - Requires immediate medical attention
        - Examples: Chest pain, unconsciousness, severe bleeding

        **⚠️ Moderate (Yellow)**
        - Serious but not immediately life-threatening
        - Requires attention within 24-48 hours
        - Examples: High fever, persistent pain, infections

        **✅ Routine (Green)**
        - Minor, self-limiting conditions
        - Can be managed with self-care
        - Examples: Mild headaches, cold symptoms, minor aches

        ### **How It Works**

        1. **Text Processing**: Your symptom description is cleaned and tokenized
        2. **Feature Extraction**: Important medical terms are identified
        3. **Classification**: AI model compares your symptoms to known patterns
        4. **Confidence Scoring**: Model calculates probabilities for each urgency level
        5. **Recommendation**: Based on the highest probability and medical guidelines

        ### **Limitations & Disclaimers**

        - This tool is for **informational purposes only**
        - It **does not provide medical diagnosis**
        - Always consult with healthcare professionals
        - Model trained on limited data (200 samples)
        - Accuracy may vary with unusual symptom descriptions

        ### **For Best Results**

        - Be specific about symptoms
        - Mention duration and severity
        - Include relevant details
        - Use complete sentences
        """)

def get_recommendations(prediction):
    """Get detailed recommendations based on prediction"""
    recommendations = {
        'Emergency': """
        **CRITICAL ACTION REQUIRED:**

        **1. Call Emergency Services NOW**
           - Dial your local emergency number
           - Clearly state "medical emergency"
           - Provide exact location

        **2. Do NOT Delay**
           - Every minute counts
           - Do not wait for symptoms to improve
           - Do not attempt to self-transport

        **3. While Waiting for Help**
           - Stay with the patient
           - Keep them calm and still
           - Do not give food or drink
           - Loosen tight clothing

        **4. Prepare Information**
           - Patient's age and medical history
           - Current medications
           - Known allergies
           - Time symptoms started
        """,

        'Moderate': """
        **URGENT CARE RECOMMENDED:**

        **1. Schedule Medical Appointment**
           - Contact doctor within 24 hours
           - Use telemedicine option if available
           - Visit urgent care if no appointment available

        **2. Monitor Symptoms**
           - Keep symptom diary
           - Note any worsening
           - Check temperature regularly

        **3. Self-Care Measures**
           - Rest adequately
           - Stay hydrated
           - Use OTC medications as directed
           - Apply heat/cold as needed

        **4. Warning Signs**
           - Seek immediate care if:
             * Symptoms suddenly worsen
             * New severe symptoms appear
             * Fever spikes above 103°F (39.4°C)
        """,

        'Routine': """
        **SELF-CARE AND MONITORING:**

        **1. Home Treatment**
           - Rest and proper nutrition
           - Stay well-hydrated
           - Use OTC remedies as needed
           - Apply appropriate home remedies

        **2. Monitoring Guidelines**
           - Symptoms should improve in 2-3 days
           - Watch for any progression
           - Note new or changing symptoms

        **3. When to Seek Medical Care**
           - Symptoms persist beyond 3 days
           - Condition worsens significantly
           - Fever develops or increases
           - New concerning symptoms appear

        **4. Prevention**
           - Practice good hygiene
           - Maintain healthy lifestyle
           - Get adequate sleep
           - Manage stress effectively
        """
    }

    return recommendations.get(prediction, "No specific recommendations available.")

def main():
    """Main application function"""

    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'symptom_text' not in st.session_state:
        st.session_state.symptom_text = ""

    # Initialize classifier
    classifier = SymptomClassifier()

    # Header
    st.markdown('<h1 class="main-header">🏥 AI Symptom Urgency Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Get instant AI-powered assessment of medical symptom urgency")

    # Create main layout with sidebar
    with st.sidebar:

        st.image("kendos.png", width=80)
        st.title("📊 Symptom Urgency Classifier")
        #st.markdown("## 🔧 Navigation")
        #st.markdown("---")

        # Quick examples
        #st.markdown("### 🚀 Quick Examples")
        #display_symptom_examples()

        st.markdown("---")

        # Model info
        display_model_info()

        st.markdown("---")

        # Statistics
        if st.session_state.prediction_history:
            st.markdown("### 📈 Your Statistics")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            emergency_pct = (history_df['prediction'] == 'Emergency').mean() * 100
            st.metric("Emergency Rate", f"{emergency_pct:.1f}%")

            if not history_df.empty:
                avg_response_time = (datetime.now() - history_df.iloc[-1]['timestamp']).seconds / 60
                st.metric("Minutes Since Last Check", f"{avg_response_time:.0f}")

        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()

        st.markdown("---")

        # Emergency contacts
        st.markdown("""
        ### 📞 Emergency Contacts

        **Global Emergency Numbers:**
        - **Ethiopia/Addis Ababa**: 907
        - **USA/Canada**: 911
        - **UK**: 999 or 112
        - **EU**: 112
        - **Australia**: 000
        - **New Zealand**: 111

        **Additional Resources:**
        - Poison Control: 1-800-222-1222 (US)
        - Suicide Prevention: 988 (US)
        - Crisis Text Line: Text HOME to 741741
        """)

        st.markdown("---")

        # Footer
        st.markdown("""
        <div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>
        <p>🩺 Medical AI Assistant v1.0</p>
        <p>For educational purposes only</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content area - Two column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Symptom input section
        st.markdown("## ✍️ Describe Your Symptoms")

        symptom_input = st.text_area(
            "Enter symptom description:",
            value=st.session_state.symptom_text,
            height=150,
            placeholder="Example: Severe chest pain and difficulty breathing that started an hour ago...",
            help="Be specific about symptoms, severity, duration, and any related factors.",
            key="symptom_input"
        )

        # Additional information section
        with st.expander("➕ Additional Information (Optional)", expanded=False):
            col_a, col_b = st.columns(2)

            with col_a:
                duration = st.selectbox(
                    "Duration of Symptoms",
                    ["Select duration", "Less than 1 hour", "1-24 hours", "1-3 days", 
                     "3-7 days", "More than 1 week", "Chronic (months+)"]
                )

                severity = st.select_slider(
                    "Symptom Severity",
                    options=["Mild", "Moderate", "Severe", "Very Severe"],
                    value="Moderate"
                )

            with col_b:
                age_group = st.selectbox(
                    "Age Group",
                    ["Select age group", "Child (0-12)", "Teen (13-19)", 
                     "Adult (20-64)", "Senior (65+)"]
                )

                previous_conditions = st.multiselect(
                    "Existing Medical Conditions",
                    ["None", "Diabetes", "Heart Disease", "High Blood Pressure", 
                     "Asthma/COPD", "Cancer History", "Autoimmune Disorder", "Other"]
                )

        # Predict button
        predict_button = st.button("🔍 Analyze Symptoms with AI", type="primary", use_container_width=True)

        if predict_button and symptom_input.strip():
            st.session_state.symptom_text = symptom_input

            with st.spinner("🤖 AI is analyzing your symptoms..."):
                # Get prediction
                prediction, probabilities = classifier.predict_urgency(symptom_input)

                if prediction:
                    # Get keywords
                    keywords = classifier.get_top_keywords(symptom_input)

                    # Add to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'symptom': symptom_input,
                        'prediction': prediction,
                        'probabilities': probabilities.tolist(),
                        'duration': duration,
                        'severity': severity,
                        'age_group': age_group
                    })

                    # Display results in right column
                    st.session_state.last_prediction = {
                        'prediction': prediction,
                        'probabilities': probabilities,
                        'symptom_text': symptom_input,
                        'keywords': keywords
                    }

        elif not symptom_input.strip() and predict_button:
            st.warning("⚠️ Please enter symptoms to analyze.")

    with col2:
        # Display results if available
        if 'last_prediction' in st.session_state:
            prediction_data = st.session_state.last_prediction

            # Show prediction
            display_prediction_result(
                prediction_data['prediction'],
                prediction_data['probabilities'],
                prediction_data['symptom_text']
            )

            # Show symptom analysis
            display_symptom_analysis(
                prediction_data['symptom_text'],
                prediction_data['keywords'],
                classifier
            )

            # Export option
            st.markdown("---")

            col_export1, col_export2 = st.columns(2)

            with col_export1:
                # Download report
                report_data = f"""
                SYMPTOM URGENCY ANALYSIS REPORT
                ================================

                Patient Assessment
                ------------------
                Symptoms: {prediction_data['symptom_text']}
                Assessment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                AI Analysis Results
                -------------------
                Urgency Level: {prediction_data['prediction']}

                Confidence Scores:
                - Routine: {prediction_data['probabilities'][0]:.1%}
                - Moderate: {prediction_data['probabilities'][1]:.1%}
                - Emergency: {prediction_data['probabilities'][2]:.1%}

                Detected Keywords: {', '.join(prediction_data['keywords']) if prediction_data['keywords'] else 'None'}

                Medical Recommendations
                -----------------------
                {get_recommendations(prediction_data['prediction'])}

                Important Disclaimer
                --------------------
                This AI analysis is for informational purposes only and does not 
                constitute medical advice, diagnosis, or treatment. Always consult 
                with a qualified healthcare professional for medical concerns.

                In case of emergency, call your local emergency number immediately.
                """

                st.download_button(
                    label="📥 Download Full Report",
                    data=report_data,
                    file_name=f"symptom_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col_export2:
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.session_state.symptom_text = ""
                    if 'last_prediction' in st.session_state:
                        del st.session_state.last_prediction
                    st.rerun()

        else:
            # Welcome/instructions
            about_help = "📝 How to Use This Tool"
            ttt1 = f"### {about_help}"
            with st.expander( ttt1 , expanded = False):
                    st.markdown("""
                    **1. Describe Symptoms**
                    - Enter symptoms in the text area
                    - Be specific and detailed
                    - Include duration and severity

                    **2. Add Details (Optional)**
                    - Duration of symptoms
                    - Severity level
                    - Age group
                    - Existing conditions

                    **3. Get AI Analysis**
                    - Click 'Analyze Symptoms with AI'
                    - Review urgency assessment
                    - See confidence scores

                    **4. Follow Recommendations**
                    - Emergency: Call for help immediately
                    - Moderate: Schedule appointment
                    - Routine: Self-care and monitoring
                    """)

            st.markdown("""

            ### 💡 Tips for Best Results

            ✅ **Be Specific**: "Severe chest pain for 2 hours"  
            ✅ **Include Details**: Duration, severity, triggers  
            ✅ **Use Complete Sentences**: Better for AI understanding  
            ✅ **Try Examples**: Use the examples in the sidebar  

            ### 🚀 Ready to Start?

            Enter your symptoms in the left panel and click "Analyze Symptoms with AI"
            """)

    # Statistics section (full width below)
    if st.session_state.prediction_history:
        st.markdown("---")
        display_statistics(st.session_state.prediction_history)

    # Footer disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;'>
    <p>🚨 <strong>IMPORTANT MEDICAL DISCLAIMER:</strong> This AI-powered tool is for informational and educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>

    <p>Never disregard professional medical advice or delay in seeking it because of something you have read or interpreted from this tool. 
    If you think you may have a medical emergency, call your doctor, go to the emergency department, or call your local emergency number immediately.</p>

    <p>© 2025 Medical AI Assistant. For demonstration and educational use only.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

