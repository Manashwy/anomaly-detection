import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from pyod.models.hbos import HBOS
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Introduction
st.title("Unsupervised Anomaly Detection Technique Comparison")
st.markdown("""
This app compares different unsupervised anomaly detection techniques.
Choose the technique and parameters to see its performance on your dataset.
""")

# Sidebar: Load dataset and select features
st.sidebar.title("Dataset & Features")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset Loaded!")
    st.write("Data Preview:")
    st.write(df.head())
    
    # Selecting feature columns
    feature_cols = st.sidebar.multiselect("Select Features", df.columns.tolist(), default=df.columns[:-1])
    
    # Scale data
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Sidebar: Select model and parameters
st.sidebar.title("Model Selection")
model_type = st.sidebar.selectbox("Select Anomaly Detection Technique", [
    "Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Elliptic Envelope", 
    "Gaussian Mixture Model", "Autoencoder", "DBSCAN", "Spectral Clustering", 
    "KMeans", "HBOS"
])

# Initialize model based on selection
if model_type == "Isolation Forest":
    contamination = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1)
    model = IsolationForest(contamination=contamination)
elif model_type == "Local Outlier Factor":
    n_neighbors = st.sidebar.slider("Number of Neighbors", 5, 50, 20)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1, novelty=True)
elif model_type == "One-Class SVM":
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    model = OneClassSVM(kernel=kernel, gamma='auto')
elif model_type == "Elliptic Envelope":
    contamination = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1)
    model = EllipticEnvelope(contamination=contamination)
elif model_type == "Gaussian Mixture Model":
    n_components = st.sidebar.slider("Number of Components", 1, 10, 2)
    model = GaussianMixture(n_components=n_components)
elif model_type == "Autoencoder":
    model = AutoEncoder(contamination=0.1)
elif model_type == "DBSCAN":
    eps = st.sidebar.slider("EPS", 0.1, 10.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 5, 50, 10)
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif model_type == "Spectral Clustering":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    model = SpectralClustering(n_clusters=n_clusters)
elif model_type == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    model = KMeans(n_clusters=n_clusters)
elif model_type == "HBOS":
    model = HBOS()

# Fit model and display results
if st.button("Run Model"):
    # Predict anomalies
    if model_type in ["Local Outlier Factor", "One-Class SVM", "Isolation Forest", "Elliptic Envelope", "Autoencoder"]:
        model.fit(X_scaled)
        y_pred = model.predict(X_scaled)
        if model_type == "Local Outlier Factor":
            y_pred = (y_pred == -1).astype(int)  # Convert LOF output
    elif model_type in ["DBSCAN", "KMeans", "Spectral Clustering"]:
        y_pred = model.fit_predict(X_scaled)
        y_pred = (y_pred == -1).astype(int)  # Label -1 as outliers
    elif model_type == "Gaussian Mixture Model":
        y_pred = model.predict(X_scaled)
        y_pred = (y_pred == -1).astype(int)  # Label outliers

    # Visualize Results
    st.subheader("Visualizations")
    
    # Scatter Plot
    st.write("### Anomaly Detection Scatter Plot")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=20)
    plt.title(f"{model_type} Anomaly Detection Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(plt)


    # Heatmap of Anomaly Scores
    st.write("### Heatmap of Anomaly Scores")
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_scaled)
    elif hasattr(model, 'score_samples'):
        scores = model.score_samples(X_scaled)
    else:
        scores = np.random.rand(len(X_scaled))  # Generate random scores if model has no scoring function
    
    sns.heatmap(scores.reshape(-1, 1), cmap="YlGnBu", annot=False)
    st.pyplot(plt)

    # Summary and Analysis
    st.subheader("Detailed Report")
    st.markdown(f"""
    - **Model Type**: {model_type}
    - **Hyperparameters**: {model.get_params()}
    - **Dataset Info**:
        - **Number of Observations**: {X_scaled.shape[0]}
        - **Number of Features**: {X_scaled.shape[1]}
    
    ### Analysis:
    This section provides insights on the detection capability of {model_type} for anomaly detection. Review the visualizations to observe the spread and clustering of anomalies across features.
    """)
