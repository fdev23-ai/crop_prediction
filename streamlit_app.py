import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🌾 Global Crop Production Predictor", layout="centered")

# Title
st.title("🌍 Global Crop Production Predictor")
st.markdown("Predict total crop production using **Area**, **Yield**, **Year**, **Crop**, and **Country** based on FAOSTAT data.")

# Load and preprocess the data
@st.cache_data
def load_data():
    df_raw = pd.read_csv("crop_production.csv")  # Replace with your FAOSTAT-style dataset

    # Select important columns
    df = df_raw[['Area', 'Item', 'Element', 'Year', 'Value']]

    # Pivot to make one row per Country-Crop-Year
    df_pivot = df.pivot_table(
        index=['Area', 'Item', 'Year'],
        columns='Element',
        values='Value'
    ).reset_index()

    # Rename for convenience
    df_pivot.columns.name = None
    df_pivot.rename(columns={
        'Area': 'Country',
        'Item': 'Crop',
        'Area harvested': 'Area',
        'Yield': 'Yield',
        'Production': 'Production'
    }, inplace=True)

    # Drop rows with missing values
    df_pivot.dropna(subset=['Area', 'Yield', 'Production'], inplace=True)

    # Encode categorical variables
    df_pivot['Crop'] = df_pivot['Crop'].astype('category')
    df_pivot['Crop_Code'] = df_pivot['Crop'].cat.codes

    df_pivot['Country'] = df_pivot['Country'].astype('category')
    df_pivot['Country_Code'] = df_pivot['Country'].cat.codes

    return df_pivot

# Load data
df = load_data()

# Display Data
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Features and Target
X = df[['Country_Code', 'Crop_Code', 'Year', 'Area', 'Yield']]
y = df['Production']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Evaluation")
st.write(f"**MAE:** {mae:,.2f}")
st.write(f"**MSE:** {mse:,.2f}")
st.write(f"**RMSE:** {rmse:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Scatter Plot
st.subheader("📉 Actual vs Predicted")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Production")
ax.set_ylabel("Predicted Production")
st.pyplot(fig)

# Prediction UI
st.subheader("🔍 Try a Prediction")

# Dropdowns for Country and Crop
country_list = df['Country'].cat.categories.tolist()
crop_list = df['Crop'].cat.categories.tolist()

selected_country = st.selectbox("🌍 Select Country", country_list)
selected_crop = st.selectbox("🌾 Select Crop", crop_list)
year = st.slider("📅 Select Year", min_value=int(df['Year'].min()), max_value=2030, value=2022)
area = st.number_input("🧱 Area Harvested (ha)", value=10000.0, step=1000.0)
yield_val = st.number_input("🌱 Yield (kg/ha)", value=2500.0, step=100.0)

# Prepare input for prediction
input_data = pd.DataFrame([[
    country_list.index(selected_country),
    crop_list.index(selected_crop),
    year,
    area,
    yield_val
]], columns=['Country_Code', 'Crop_Code', 'Year', 'Area', 'Yield'])

# Predict
predicted = model.predict(input_data)[0]

st.success(f"📦 Estimated Crop Production: **{predicted:,.2f} tons**")
