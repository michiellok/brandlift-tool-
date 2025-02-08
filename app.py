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
budgets = np.random.randint(5000, 50000, size=200)
exposed_lift = np.random.uniform(0.01, 0.15, size=200)
non_exposed_lift = np.random.uniform(0.005, 0.08, size=200)
media_channel = np.random.choice(campaigns, size=200)
attention_scores = np.random.uniform(0.1, 1.0, size=200)
audience_segment = np.random.choice(audience_segments, size=200)
conversion_likelihood = np.random.uniform(0.01, 0.5, size=200)

data = pd.DataFrame({
    'Campaign': media_channel,
    'Budget': budgets,
    'Exposed_Lift': exposed_lift,
    'Non_Exposed_Lift': non_exposed_lift,
    'Lift_Difference': exposed_lift - non_exposed_lift,
    'Attention_Score': attention_scores,
    'Audience_Segment': audience_segment,
    'Conversion_Likelihood': conversion_likelihood
})

# Train Machine Learning Model to Predict Brand Lift
X = data[['Budget', 'Attention_Score', 'Conversion_Likelihood']]
y = data['Lift_Difference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit Dashboard UI
st.title("Brand Impact & Audience Intelligence Dashboard")
st.write("Predicting Brand Lift and Optimizing Media Selection")

# Model Performance Display
st.metric(label="Model Mean Absolute Error (MAE)", value=f"{mae:.4f}")

# Data Display
st.subheader("Top Performing Campaigns")
data_sorted = data.sort_values(by='Lift_Difference', ascending=False)
st.dataframe(data_sorted.head(10))

# Filtered View
st.subheader("Filter by Media Channel and Audience Segment")
selected_channel = st.selectbox("Select Campaign Type:", options=campaigns)
selected_audience = st.selectbox("Select Audience Segment:", options=audience_segments)
st.dataframe(data[(data['Campaign'] == selected_channel) & (data['Audience_Segment'] == selected_audience)])

# Suggested Campaign Activation
st.subheader("Recommended Media Plan")
suggested_campaign = data_sorted.head(1)
st.write("Based on the highest impact and attention score, we recommend the following:")
st.dataframe(suggested_campaign[['Campaign', 'Budget', 'Lift_Difference', 'Attention_Score', 'Audience_Segment', 'Conversion_Likelihood']])
st.write("This campaign can be activated in the DSP for optimal performance.")
