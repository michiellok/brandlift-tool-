import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
    'Frequency': frequency
})

# Train Machine Learning Model to Predict Brand Lift
X = data[['Budget', 'Attention_Score', 'Conversion_Likelihood', 'CPM', 'Frequency']]
y = data['Lift_Difference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit Dashboard UI
st.title("AI-Powered Media Planning Dashboard")
st.write("Optimize your media investment based on impact, attention, and audience insights")

# Model Performance Display
st.metric(label="Model Mean Absolute Error (MAE)", value=f"{mae:.4f}")

# Advertiser Campaign Goal Input
st.subheader("Define Campaign Objectives")
campaign_name = st.text_input("Campaign Name")
selected_objective = st.selectbox("Select Campaign Objective:", options=objectives)
selected_audience_goal = st.selectbox("Select Audience Target:", options=audience_segments)
selected_budget = st.slider("Select Budget Range:", min_value=5000, max_value=50000, step=5000)
selected_frequency = st.slider("Select Desired Frequency:", min_value=1, max_value=10, step=1)
selected_cpm_cap = st.slider("Select CPM Cap:", min_value=5, max_value=50, step=1)

# AI-Powered Media Recommendation
st.subheader("AI-Generated Media Plan")
predicted_campaign = data[(data['Audience_Segment'] == selected_audience_goal) &
                          (data['Budget'] <= selected_budget) &
                          (data['Frequency'] <= selected_frequency) &
                          (data['CPM'] <= selected_cpm_cap)]
predicted_campaign = predicted_campaign.sort_values(by=['Lift_Difference', 'Attention_Score', 'Conversion_Likelihood'], ascending=False).head(3)

if not predicted_campaign.empty:
    st.write("Based on your objectives, we recommend the following media plan:")
    st.dataframe(predicted_campaign[['Campaign', 'Budget', 'Lift_Difference', 'Attention_Score', 'Audience_Segment', 'Conversion_Likelihood', 'CPM', 'Frequency']])
    st.write("This plan can be fine-tuned before activation in the DSP.")
else:
    st.write("No optimal campaign found based on current constraints. Try adjusting your inputs.")

# Manual Adjustments
st.subheader("Manual Media Plan Adjustments")
st.write("Modify media allocation before finalizing the plan")
selected_channel_adjust = st.selectbox("Adjust Media Channel", options=campaigns)
adjusted_budget = st.slider("Adjust Budget:", min_value=5000, max_value=50000, step=5000)
adjusted_frequency = st.slider("Adjust Frequency:", min_value=1, max_value=10, step=1)
adjusted_cpm = st.slider("Adjust CPM:", min_value=5, max_value=50, step=1)

st.write(f"Your manually adjusted media allocation: {selected_channel_adjust} - Budget: {adjusted_budget}, Frequency: {adjusted_frequency}, CPM: {adjusted_cpm}")

# DSP Activation Simulation
st.subheader("Activate in DSP")
if st.button("Generate DSP Export File"):
    export_data = pd.DataFrame({
        'Campaign Name': [campaign_name],
        'Objective': [selected_objective],
        'Audience': [selected_audience_goal],
        'Budget': [adjusted_budget],
        'Frequency': [adjusted_frequency],
        'CPM': [adjusted_cpm],
        'Media Channel': [selected_channel_adjust]
    })
    export_data.to_csv("media_plan.csv", index=False)
    st.success("Your media plan has been exported for DSP activation!")
