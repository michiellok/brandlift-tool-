import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Streamlit UI Config
st.set_page_config(page_title="Brand Impact & Media Planner", layout="wide")

# Generate Simulated Dummy Data
np.random.seed(42)
campaigns = ['CTV', 'DOOH', 'Social', 'Display', 'Retail Media']
audience_segments = ['Young Adults', 'High Income', 'Gamers', 'Parents', 'Tech Enthusiasts']
objectives = ['Awareness', 'Consideration', 'Preference', 'Intent']
budgets = np.random.randint(5000, 50000, size=200)
exposed_lift = np.random.uniform(0.01, 0.15, size=200)
non_exposed_lift = np.random.uniform(0.005, 0.08, size=200)
media_channel = np.random.choice(campaigns, size=200)
attention_scores = np.random.uniform(0.1, 1.0, size=200)
audience_segment = np.random.choice(audience_segments, size=200)
conversion_likelihood = np.random.uniform(0.01, 0.5, size=200)
cpm = np.random.uniform(5, 50, size=200)
frequency = np.random.randint(1, 10, size=200)
campaign_duration = np.random.randint(7, 90, size=200)  # Campaign duration in days

# Benchmark Data (Simulated Industry Averages)
benchmark_data = {
    'Campaign': campaigns,
    'Avg_Brand_Lift': [0.08, 0.06, 0.07, 0.05, 0.09],
    'Avg_Attention_Score': [0.7, 0.65, 0.75, 0.6, 0.8],
    'Avg_Conversion_Likelihood': [0.15, 0.12, 0.18, 0.1, 0.2]
}
benchmark_df = pd.DataFrame(benchmark_data)

data = pd.DataFrame({
    'Campaign': media_channel,
    'Budget': budgets,
    'Exposed_Lift': exposed_lift,
    'Non_Exposed_Lift': non_exposed_lift,
    'Lift_Difference': exposed_lift - non_exposed_lift,
    'Attention_Score': attention_scores,
    'Audience_Segment': audience_segment,
    'Conversion_Likelihood': conversion_likelihood,
    'CPM': cpm,
    'Frequency': frequency,
    'Campaign_Duration': campaign_duration
})

# Train Machine Learning Model (Random Forest for better accuracy)
X = data[['Budget', 'Attention_Score', 'Conversion_Likelihood', 'CPM', 'Frequency', 'Campaign_Duration']]
y = data['Lift_Difference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit Dashboard UI
st.title("Brand Impact & Media Planning Dashboard")
st.write("Maximize your media investment with AI-driven insights and optimization.")

# Tabs for structured navigation
tab1, tab2, tab3, tab4 = st.tabs(["📌 Campaign Setup", "📊 AI Media Plan", "📈 Insights & Benchmarking", "🔍 Deep-Dive Analysis"])

# Insights & Benchmarking Tab
with tab3:
    st.header("📈 Brand Uplift Analysis vs. Benchmark")
    selected_media = st.selectbox("Select Media Channel", options=campaigns)
    brand_lift_adjustment = st.slider("Adjust Brand Uplift", -0.05, 0.05, 0.0, step=0.01)
    
    # Get actual and benchmark brand uplift
    actual_lift = data[data['Campaign'] == selected_media]['Lift_Difference'].mean()
    benchmark_lift = benchmark_df[benchmark_df['Campaign'] == selected_media]['Avg_Brand_Lift'].values[0]
    adjusted_lift = actual_lift + brand_lift_adjustment
    
    # Display uplift values
    st.metric("Actual Brand Uplift", f"{actual_lift:.2%}")
    st.metric("Benchmark Brand Uplift", f"{benchmark_lift:.2%}")
    st.metric("Adjusted Brand Uplift", f"{adjusted_lift:.2%}")
    
    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(x=['Actual', 'Benchmark', 'Adjusted'], y=[actual_lift, benchmark_lift, adjusted_lift], palette=['blue', 'red', 'green'])
    plt.ylabel("Brand Uplift")
    plt.title(f"Brand Uplift Comparison for {selected_media}")
    st.pyplot(fig)
    
    # AI Recommendation
    if adjusted_lift > benchmark_lift:
        st.success(f"🚀 AI suggests increasing investment in **{selected_media}**, as adjusted uplift ({adjusted_lift:.2%}) is above industry benchmark.")
    else:
        st.warning(f"⚠️ AI suggests reviewing investment in **{selected_media}**, as adjusted uplift ({adjusted_lift:.2%}) is below industry benchmark.")







