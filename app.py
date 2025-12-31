import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💸",
    layout="wide"
)

# --- 2. Custom CSS ---
st.markdown("""
<style>
    /* Main Layout */
    .main { background-color: #f5f7f9; }
    h1 { color: #1E3D59; font-family: 'Helvetica', sans-serif; }
    
    /* Green Card (Predicted Salary) */
    .metric-card-green {
        background-color: #d4edda; border: 1px solid #c3e6cb;
        padding: 20px; border-radius: 10px; color: #155724;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-green h2 { margin: 0; font-size: 18px; font-weight: normal; color: #155724; }
    .metric-card-green h1 { margin: 0; font-size: 40px; font-weight: bold; color: #155724; }
    
    /* Yellow Card (Typical Range) */
    .metric-card-yellow {
        background-color: #fff3cd; border: 1px solid #ffeeba;
        padding: 20px; border-radius: 10px; color: #856404;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card-yellow h2 { margin: 0; font-size: 18px; font-weight: normal; color: #856404; }
    .metric-card-yellow h1 { margin: 0; font-size: 30px; font-weight: bold; color: #856404; }
    
    /* Green Predict Button */
    div.stButton > button {
        background-color: #28a745; color: white; font-size: 18px; font-weight: bold;
        height: 50px; width: 100%; border-radius: 8px; border: none; transition: all 0.3s;
    }
    div.stButton > button:hover { background-color: #218838; transform: scale(1.02); }
    
    /* Section Headers */
    .section-header {
        font-size: 20px; font-weight: bold; color: #333; margin-top: 20px; margin-bottom: 10px;
        border-bottom: 2px solid #28a745; padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Load Resources ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('salary_model_final.pkl')
        metadata = joblib.load('app_metadata.pkl')
        
        # Patch for old metadata files (Safety Check)
        if "Internship" not in metadata['experience_map']:
            metadata['experience_map']['Internship'] = 0
            
        return model, metadata
    except FileNotFoundError:
        return None, None

model, metadata = load_data()

if model is None:
    st.error("⚠️ Files missing! Please run the export code in your notebook.")
    st.stop()

# --- 4. Prediction Logic ---
def extract_skills(df, text_col):
    keywords = ['python', 'sql', 'java', 'aws', 'azure', 'spark', 'react', 'kubernetes']
    for key in keywords:
        df[f'has_{key}'] = df[text_col].str.lower().apply(lambda x: 1 if key in x else 0)
    df['has_kubernetes'] = df[text_col].str.lower().apply(lambda x: 1 if 'kubernetes' in x or 'k8s' in x else df['has_kubernetes'])
    return df

def get_prediction(title, desc, loc, exp_level, remote_val):
    # Smart Mapping
    if exp_level == 0:   title_val, exp_val = 0, 1
    elif exp_level == 1: title_val, exp_val = 1, 1
    elif exp_level == 2: title_val, exp_val = 2, 3
    elif exp_level == 3: title_val, exp_val = 3, 4
    else:                title_val, exp_val = 4, 4

    input_data = pd.DataFrame({
        'title_clean': [title], 'description_clean': [desc],
        'location_group': [loc], 'experience_encoded': [exp_val],
        'remote_allowed': [1 if remote_val == "Yes" else 0],
        'text_feature': [f"{title} {desc}"], 
        'pay_period': ['YEARLY'], 'company_size': ['Unknown'], 'employment_type': ['Full-time']
    })
    
    input_data = extract_skills(input_data, 'text_feature')
    input_data['title_seniority_ordinal'] = title_val
    
    return np.expm1(model.predict(input_data)[0])

# --- 5. Sidebar & Header ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=80)
    st.markdown("### 🤖 Salary Model")
    st.info("This tool uses **XGBoost** trained on 3k+ tech job postings to estimate market value.")
    st.markdown("---")
    st.caption("© 2025 FYP Project")

st.title("💸 Salary Consultant")
st.markdown("##### 🚀 Get a data-driven salary estimate and career advice in seconds.")
st.write("")

# --- 6. Main Inputs ---
with st.container():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">1. Job Details</div>', unsafe_allow_html=True)
        
        # --- CHANGED: JOB TITLE IS NOW A DROPDOWN ---
        # We try to find 'Data Scientist' as the default startup value
        job_options = metadata.get('job_titles', ["Data Scientist (Default)"])
        
        try:
            default_ix = job_options.index("Data Scientist")
        except ValueError:
            default_ix = 0
            
        job_title = st.selectbox("Job Title", job_options, index=default_ix)
        
        # Seniority Dropdown
        order = ["Internship", "Entry Level", "Mid Level", "Senior Level", "Executive"]
        opts = [o for o in order if o in metadata['experience_map']]
        exp_label = st.selectbox("Seniority Level", opts)
        
        remote = st.radio("Remote Work?", ["No", "Yes"], horizontal=True)

    with col2:
        st.markdown('<div class="section-header">2. Location & Context</div>', unsafe_allow_html=True)
        location = st.selectbox("Location", metadata['locations'])
        description = st.text_area("Job Description / Skills", height=125, 
                                   placeholder="Paste job description here (e.g. Python, AWS required...)")
    
    st.write("")
    if st.button("✨ Analyze Salary"):
        with st.spinner("🤖 Crunching numbers & analyzing market trends..."):
            time.sleep(0.8) # UX delay
            
            # --- CALCULATIONS ---
            current_level = metadata['experience_map'][exp_label]
            pred_salary = get_prediction(job_title, description, location, current_level, remote)
            
            lower_bound = pred_salary * 0.88
            upper_bound = pred_salary * 1.12
            
            # --- RELOCATION ENGINE ---
            loc_recommendations = []
            for loc in metadata['locations']:
                if loc != location:
                    val = get_prediction(job_title, description, loc, current_level, remote)
                    loc_recommendations.append((loc, val))
            
            loc_recommendations.sort(key=lambda x: x[1], reverse=True)
            top_3_locs = loc_recommendations[:3]
            
            # --- DISPLAY RESULTS (CUSTOM CARDS) ---
            st.markdown("---")
            
            res_col1, res_col2, res_col3 = st.columns([1, 1, 1.5])
            
            with res_col1:
                st.markdown(f"""
                <div class="metric-card-green">
                    <h2>Predicted Salary</h2>
                    <h1>${pred_salary:,.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div class="metric-card-yellow">
                    <h2>Typical Range</h2>
                    <h1>${lower_bound:,.0f} - {upper_bound:,.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col3:
                input_df = pd.DataFrame({'text_feature': [f"{job_title} {description}"]})
                input_df = extract_skills(input_df, 'text_feature')
                skills = [c.replace('has_', '').title() for c in input_df.columns if input_df[c][0]==1 and c!='text_feature']
                
                if skills:
                    st.success(f"✅ Skills Detected: {', '.join(skills)}")
                else:
                    st.warning("No technical skills detected (Python, SQL, etc).")

            # --- INSIGHTS ---
            st.markdown("### 📊 Market Insights")
            chart_col, advice_col = st.columns([2, 1])
            
            with chart_col:
                st.markdown("**Career Growth Trajectory**")
                levels_map = {0: "Intern", 1: "Entry", 2: "Mid", 3: "Senior", 4: "Exec"}
                growth_data = []
                for code, name in levels_map.items():
                    val = get_prediction(job_title, description, location, code, remote)
                    growth_data.append({"Level": name, "Salary": val})
                
                st.bar_chart(pd.DataFrame(growth_data).set_index("Level"), color="#28a745")
                
            with advice_col:
                st.markdown("**💡 Relocation Tips**")
                st.write(f"Top paying cities for *{job_title}*:")
                for i, (city, pay) in enumerate(top_3_locs):
                    diff = pay - pred_salary
                    icon = "🔥" if diff > 0 else "📉"
                    color = "green" if diff > 0 else "red"
                    st.markdown(f"**{i+1}. {city}**")
                    st.markdown(f":{color}[${pay:,.0f} ({icon} ${diff:,.0f})]")

            # Download
            report = f"Role: {job_title}\nExp: {exp_label}\nLoc: {location}\n\nPrediction: ${pred_salary:,.2f}\nRange: ${lower_bound:,.0f}-${upper_bound:,.0f}"
            st.download_button("📄 Save Report", report, file_name="salary_report.txt")
