import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction Dashboard")
st.markdown("### Built with Linear Regression & Random Forest")

# ---------------------------
# Load Dataset
# ---------------------------
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

X = df.drop("Price", axis=1)
y = df["Price"]

# ---------------------------
# Train Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Train Models
# ---------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------
# Model Evaluation
# ---------------------------
lr_r2 = r2_score(y_test, lr_model.predict(X_test))
rf_r2 = r2_score(y_test, rf_model.predict(X_test))

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test)))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter House Details")

MedInc = st.sidebar.number_input("Median Income", value=3.0)
HouseAge = st.sidebar.number_input("House Age", value=20)
AveRooms = st.sidebar.number_input("Average Rooms", value=5.0)
AveBedrms = st.sidebar.number_input("Average Bedrooms", value=1.0)
Population = st.sidebar.number_input("Population", value=1000)
AveOccup = st.sidebar.number_input("Average Occupancy", value=3.0)
Latitude = st.sidebar.number_input("Latitude", value=34.0)
Longitude = st.sidebar.number_input("Longitude", value=-118.0)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Linear Regression", "Random Forest")
)

# ---------------------------
# Layout Columns
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Performance")
    st.write(f"Linear Regression R²: {lr_r2:.3f}")
    st.write(f"Random Forest R²: {rf_r2:.3f}")
    st.write(f"Linear Regression RMSE: {lr_rmse:.3f}")
    st.write(f"Random Forest RMSE: {rf_rmse:.3f}")

with col2:
    st.subheader("📈 Model Comparison Chart")
    models = ["Linear Regression", "Random Forest"]
    scores = [lr_r2, rf_r2]

    fig, ax = plt.subplots()
    ax.bar(models, scores)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Comparison")
    st.pyplot(fig)

# ---------------------------
# Prediction
# ---------------------------
if st.sidebar.button("Predict Price"):

    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])

    scaled_features = scaler.transform(features)

    if model_choice == "Linear Regression":
        prediction = lr_model.predict(scaled_features)
    else:
        prediction = rf_model.predict(scaled_features)

    final_price = prediction[0] * 100000

    st.success(f"🏡 Predicted House Price: ${final_price:,.2f}")

# ---------------------------
# Feature Importance (RF)
# ---------------------------
st.subheader("🔥 Feature Importance (Random Forest)")

importances = rf_model.feature_importances_
feature_names = X.columns

fig2, ax2 = plt.subplots()
ax2.barh(feature_names, importances)
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance")
st.pyplot(fig2)
