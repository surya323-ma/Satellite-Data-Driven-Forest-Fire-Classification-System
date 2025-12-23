import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Forest Fire Classification Dashboard",
    page_icon="🔥",
    layout="wide"
)

# ================= GENERATE SYNTHETIC MODIS DATA =================
@st.cache_data
def generate_modis_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "Brightness": np.random.uniform(300, 500, 1000),
        "Confidence": np.random.randint(20, 100, 1000),
        "Latitude": np.random.uniform(8, 37, 1000),     # India range
        "Longitude": np.random.uniform(68, 97, 1000),  # India range
        "Fire_Type": np.random.choice([0, 1, 2, 3], 1000)
    })
    return df

df = generate_modis_data()

# ================= MODEL TRAINING =================
X = df.drop("Fire_Type", axis=1)
y = df["Fire_Type"]

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X, y)

# ================= SIDEBAR =================
st.sidebar.title("🔥 Fire Detection System")
menu = st.sidebar.radio(
    "Menu",
    ["Dashboard", "EDA", "Prediction", "Fire Map", "About"]
)

# ================= DASHBOARD =================
if menu == "Dashboard":
    st.title("🔥 Forest Fire Classification Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Fire Classes", "4")
    col3.metric("Model Used", "Random Forest")

    st.markdown("---")
    st.subheader("Fire Type Distribution")

    labels = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}
    fire_counts = df["Fire_Type"].map(labels).value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=fire_counts.index, y=fire_counts.values, ax=ax)
    ax.set_xlabel("Fire Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ================= EDA =================
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ================= PREDICTION =================
elif menu == "Prediction":
    st.title("🎯 Fire Type Prediction")

    col1, col2 = st.columns(2)

    with col1:
        brightness = st.number_input("Brightness", 300.0, 500.0, 360.0)
        confidence = st.slider("Confidence", 0, 100, 60)

    with col2:
        latitude = st.number_input("Latitude", 8.0, 37.0, 22.0)
        longitude = st.number_input("Longitude", 68.0, 97.0, 78.0)

    if st.button("Predict Fire Type"):
        sample = np.array([[brightness, confidence, latitude, longitude]])
        pred = model.predict(sample)[0]

        fire_map = {
            0: "🔥 Low Fire",
            1: "🔥 Moderate Fire",
            2: "🔥 High Fire",
            3: "🔥 Extreme Fire"
        }

        st.success(f"Predicted Fire Type: **{fire_map[pred]}**")

# ================= FIRE MAP =================
elif menu == "Fire Map":
    st.title("🗺️ Forest Fire Locations Across India")

    fire_label_map = {
        0: "Low",
        1: "Moderate",
        2: "High",
        3: "Extreme"
    }

    df["Fire_Label"] = df["Fire_Type"].map(fire_label_map)

    color_map = {
        "Low": [0, 255, 0],
        "Moderate": [255, 255, 0],
        "High": [255, 165, 0],
        "Extreme": [255, 0, 0]
    }

    df["color"] = df["Fire_Label"].apply(lambda x: color_map[x])

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=22.5,
                longitude=78.9,
                zoom=4.5,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[Longitude, Latitude]",
                    get_fill_color="color",
                    get_radius=20000,
                    pickable=True,
                )
            ],
            tooltip={
                "html": "<b>Fire Type:</b> {Fire_Label}<br/>"
                        "<b>Brightness:</b> {Brightness}<br/>"
                        "<b>Confidence:</b> {Confidence}"
            }
        )
    )

# ================= ABOUT =================
else:
    st.title("ℹ️ About Project")
    st.markdown("""
    **Classification of Fire Types in India Using MODIS Satellite Data**

    - Dataset: Synthetic MODIS-style satellite data  
    - Algorithm: Random Forest Classifier  
    - Visualization: Streamlit & PyDeck  

    **Applications**
    - Forest fire monitoring  
    - Disaster risk assessment  
    - Academic & internship demonstrations  
    """)

st.markdown("---")
st.caption("🚀 Developed by Surya Omar | AI & ML Project")
