import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Config
st.set_page_config(page_title="Atlas Lions Predictor", page_icon="🇲🇦")

# 2. Load Models (Silently)
@st.cache_resource # Had l-line k-t-khalli l-app t-koun khfifa
def load_assets():
    try:
        model = joblib.load('morocco_wc_model.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, encoder
    except:
        return None, None

model, encoder = load_assets()

# 3. Sidebar UI
st.sidebar.header("📍 Match Details")
opponent = st.sidebar.selectbox("Select Opponent", ["France", "Spain", "Brazil", "Argentina", "Portugal", "USA", "Other"])
comp_type = st.sidebar.selectbox("Competition Type", ["Official", "Friendly"])
st.sidebar.divider()
st.sidebar.subheader("📈 Performance Metrics")
form_score = st.sidebar.slider("Morocco Form", 0.0, 1.0, 0.7)
goal_diff = st.sidebar.number_input("Average Goal Difference", -5.0, 5.0, 1.0)

# 4. Main UI
st.title("🦁 Atlas Lions 2026 Predictor")
st.markdown("Predicting Morocco's success based on **Historical Data**.")

if model is None:
    st.error("⚠️ Model files (.pkl) not found in the directory!")
else:
    if st.button("🚀 Predict Result"):
        # Logic
        strong_teams = ['France', 'Spain', 'Brazil', 'Argentina', 'Portugal']
        strength = 'Strong' if opponent in strong_teams else 'Medium'
        input_df = pd.DataFrame({'competition_type': [comp_type], 'opponent_strength_level': [strength]})
        
        encoded_cols = encoder.transform(input_df)
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out())
        
        features = pd.concat([
            pd.DataFrame({'form_score': [form_score], 'goal_difference': [goal_diff]}), 
            encoded_df
        ], axis=1)
        
        prob = model.predict_proba(features)[0][1]
        
        # Results Display
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Win Probability", f"{prob*100:.1f}%")
        with col2:
            if prob > 0.6:
                st.success(f"**Dima Maghrib!** Strong chance against {opponent}.")
                st.balloons()
            else:
                st.warning(f"Tough match against {opponent}.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("🚀 Developed by **Walid Arrich**")
st.sidebar.caption("Data Scientist | GazellePub")