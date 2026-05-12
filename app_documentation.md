# Fraud Detection Streamlit App — Code Documentation

# 1. Folder Structure
fraud_app/
│   app.py
│   recompute_sizes.py
│   app_documentation.md
│
├───.vscode/
│   ├── settings.json
│   └── models/
│       ├── model_fraudsage2.pt
│       ├── test_encoded.parquet
│       └── XGBoost_GNN.json
│
├───models/
│   ├── edge_index.pt
│   ├── model_fraudsage2.pt
│   ├── test_encoded.parquet
│   └── XGBoost_GNN.json
│
└───pages/
    ├── 1_Resume.py
    ├── 2_Projects.py
    ├── 3_Fraud_Detection_Demo.py
    ├── 4_About_Me.py
    └── img_dee.jpg

# 2. Main application file — app.py

import streamlit as st

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("🔍 Fraud Detection App")
st.subheader("Welcome!")

st.write("""
This application uses a hybrid **GNN + XGBoost** model to detect fraud.

Use the sidebar to navigate to:
- **Fraud Detection Demo**
""")

st.divider()

st.info("""
This system combines **Graph Neural Networks** (to capture relational patterns)
with **XGBoost** (for strong tabular performance) to produce a hybrid fraud score.
""")


# 3. Helper script — recompute_sizes.py

import pandas as pd

# Load your original train and test CSVs
train = pd.read_parquet(r"C:\Users\diana\Downloads\DTSC691_FinalProject\ieee-fraud-detection\train_clean_final.parquet")
test = pd.read_parquet(r"C:\Users\diana\Downloads\DTSC691_FinalProject\ieee-fraud-detection\test_clean_final.parquet")

cat_cols = ["P_emaildomain", "R_emaildomain", "id_30", "id_31", "DeviceInfo", "DeviceType"]

sizes = {}

for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    _, uniques = pd.factorize(combined)
    sizes[col] = len(uniques) + 1  # +1 for unknown category

print("Recomputed sizes:", sizes)

# 4. Pages
# 4.1 1_Resume.py

import streamlit as st
from pathlib import Path

st.title("📄 Resume")

# Image in the same folder as this file
image_path = Path(__file__).parent / "img_dee.jpg"

col1, col2 = st.columns([1, 3])

with col1:
    if image_path.exists():
        st.image(str(image_path), width=180)
    else:
        st.warning("Profile image not found. Make sure the file is next to 1_Resume.py")

with col2:
    st.subheader("Diana Ogendi")
    st.write("Data Scientist")
    st.write("Kirkland, WA")

st.header("Education")
st.write("""
**Master of Science — Data Science**  
Focus: Machine Learning, Big Data Systems, Graph Analytics  

**Bachelor of Science — Software Engineering**  
Focus: Systems Design, Algorithms, Distributed Computing  

**Certifications**  
- AWS Solutions Architect  
- Azure Solutions Architect  
- Azure Administrator  
""")

st.header("Work Experience")
st.write("""
### Software Engineer & Big Data Engineer — Safaricom PLC
- Designed and maintained large-scale data pipelines
- Built fraud-related data workflows with reliability and auditability
- Collaborated with cross-functional teams to deploy ML insights

### Data Analyst — BECU
- Analyzed financial transaction patterns for fraud and risk teams
- Developed dashboards and reporting tools
- Improved data quality and feature engineering pipelines
""")

# 4.2 2_Projects.py

import streamlit as st

st.title("🛠️ Projects")
st.write("""
Below are selected projects from my portfolio that highlight my experience in 
machine learning, neural networks, statistical modeling, graph analytics, and applied data science.
""")

st.divider()

# -----------------------------
# Fraud Detection App
# -----------------------------
st.subheader("🔐 Fraud Detection App — Hybrid GNN + XGBoost")
st.write("""
**Tech Stack:** Python | PyTorch Geometric | XGBoost | Streamlit | Pandas  

A full end-to-end fraud detection system combining **Graph Neural Networks (GNNs)** for relational 
embeddings with **XGBoost** for high-performance tabular classification.

**Key Features**
- Real-time fraud scoring in a polished Streamlit UI  
- Threshold tuner for analyst-friendly decisioning  
- Stable feature engineering and reproducible pipelines  
- Hybrid model architecture (GNN embeddings + XGBoost classifier)  
- Clean transaction lookup and probability visualization  

**Outcome:**  
Delivered a production-style fraud detection demo showcasing graph-based modeling, hybrid ML architectures, 
and interactive UI deployment.
""")

st.divider()

# -----------------------------
# Mushroom Classification
# -----------------------------
st.subheader("🍄 Mushroom Classification (Neural Networks)")
st.write("""
**Tech Stack:** Python | TensorFlow/Keras | UCI Mushroom Dataset  

**Analyzed Mushroom Edibility**
- Built a full preprocessing pipeline using one-hot encoding and label encoding  
- Prepared the UCI mushroom dataset for neural network modeling  

**Developed Neural Network Models**
- Trained a baseline sequential neural network  
- Trained a PCA-optimized neural network (95% variance retained)  
- Compared accuracy, dimensionality reduction effects, and training efficiency  

**Evaluated Model Performance**
- Used Scikit-Learn’s ConfusionMatrixDisplay for interpretability  
""")

st.divider()

# -----------------------------
# Student Performance Analysis
# -----------------------------
st.subheader("📚 Student Performance Analysis")
st.write("""
**Tech Stack:** Python | Jupyter Notebook | Portuguese School System Dataset  

**Exploratory Data Analysis**
- Investigated demographic, behavioral, and academic factors influencing student success  

**Predictive Modeling**
- Built Linear Regression, Support Vector Machines, and Lasso Regression models  
- Forecasted student grades using prior performance and behavioral indicators  

**Insights**
- Identified key predictors of academic outcomes  
- Evaluated model performance across multiple metrics  
""")

st.divider()

# -----------------------------
# Health Analytics (BRFSS)
# -----------------------------
st.subheader("🏥 Health Analytics (BRFSS Data)")
st.write("""
**Tech Stack:** R | RStudio | CDC BRFSS Dataset  

**Data Exploration**
- Analyzed the CDC’s Behavioral Risk Factor Surveillance System (BRFSS) dataset  

**Transformations & Insights**
- Explored health behaviors, chronic conditions, and preventive service usage  
- Produced interpretable visualizations and summaries for public health insights  
""")

# 4.3 3_Fraud_Detection_Demo.py

import os
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

st.title("🔍 Fraud Detection Demo")


st.sidebar.subheader("Decision Threshold")

threshold = st.sidebar.slider(
    "Fraud Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

# -----------------------------
#  FraudSAGE Model Definition
# -----------------------------
class FraudSAGE(nn.Module):
    def __init__(self, num_numeric_features, sizes):
        super().__init__()

        self.emb_p = nn.Embedding(sizes["P_emaildomain"], 8)
        self.emb_r = nn.Embedding(sizes["R_emaildomain"], 8)
        self.emb_id30 = nn.Embedding(sizes["id_30"], 4)
        self.emb_id31 = nn.Embedding(sizes["id_31"], 8)
        self.emb_dev = nn.Embedding(sizes["DeviceInfo"], 8)
        self.emb_dev_type = nn.Embedding(sizes["DeviceType"], 2)

        total_emb_dim = 38  # 8+8+4+8+8+2

        self.conv1 = SAGEConv(num_numeric_features + total_emb_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = SAGEConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 2)

    def forward(self, x_numeric, x_cat, edge_index):
        e_p = self.emb_p(x_cat[:, 0])
        e_r = self.emb_r(x_cat[:, 1])
        e_30 = self.emb_id30(x_cat[:, 2])
        e_31 = self.emb_id31(x_cat[:, 3])
        e_dev = self.emb_dev(x_cat[:, 4])
        e_dev_type = self.emb_dev_type(x_cat[:, 5])

        x = torch.cat([x_numeric, e_p, e_r, e_30, e_31, e_dev, e_dev_type], dim=1)

        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()

        return self.out(x)


# -----------------------------
#  Paths
# -----------------------------
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PAGES_DIR)

def abs_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


# -----------------------------
#  Load Test Data
# -----------------------------
@st.cache_data
def load_test_data():
    df = pd.read_parquet(abs_path("models", "test_encoded.parquet"))
    return df


# -----------------------------
#  Add Missing Engineered Feature
# -----------------------------
def add_missing_feature(df):
    if "missing_feature" not in df.columns:
        df["missing_feature"] = 0.0
    return df


# -----------------------------
#  Load GNN Model
# -----------------------------
@st.cache_resource
def load_gnn():
    sizes = {
        "P_emaildomain": 10,
        "R_emaildomain": 10,
        "id_30": 6,
        "id_31": 13,
        "DeviceInfo": 11,
        "DeviceType": 3
    }

    NUM_NUMERIC_FEATURES = 434  

    model = FraudSAGE(NUM_NUMERIC_FEATURES, sizes)
    state_dict = torch.load(abs_path("models", "model_fraudsage2.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -----------------------------
#  Load XGBoost Model
# -----------------------------
@st.cache_resource
def load_xgb():
    booster = xgb.Booster()
    booster.load_model(abs_path("models", "XGBoost_GNN.json"))
    return booster


# -----------------------------
#  Prepare Inputs
# -----------------------------
cat_cols = ["P_emaildomain", "R_emaildomain", "id_30", "id_31", "DeviceInfo", "DeviceType"]

def build_numeric_list(df):
    return [
        col for col in df.columns
        if col not in cat_cols and col != "TransactionID"
    ]


def prepare_graph_inputs(df, row_idx, numeric_list):
    x_numeric = torch.tensor(
        df.loc[row_idx, numeric_list].astype("float32").values.reshape(1, -1),
        dtype=torch.float32
    )

    # Force categorical columns to int64
    x_cat_vals = df.loc[row_idx, cat_cols].astype("int64").values

    x_cat = torch.tensor(
        x_cat_vals.reshape(1, -1),
        dtype=torch.long
    )

    return x_numeric, x_cat


# -----------------------------
#  UI + Inference
# -----------------------------

test_df = load_test_data()
test_df = add_missing_feature(test_df)

# Clean TransactionID dtype (fixes .0000000005 issue)
test_df["TransactionID"] = test_df["TransactionID"].astype("int64")

TRAIN_NUMERIC_FEATURE_LIST = build_numeric_list(test_df)
assert len(TRAIN_NUMERIC_FEATURE_LIST) == 434

gnn_model = load_gnn()
xgb_model = load_xgb()

# Clean dropdown display
transaction_ids = test_df["TransactionID"].astype(str).tolist()
selected_id = int(st.selectbox("Select Transaction ID", transaction_ids))

# Map back to row index
row_idx = test_df.index[test_df["TransactionID"] == selected_id][0]


x_numeric, x_cat = prepare_graph_inputs(test_df, row_idx, TRAIN_NUMERIC_FEATURE_LIST)

# Dummy self-loop edge_index for single-node inference
dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# Capture GNN embedding
bn2_cache = {}
def save_bn2_output(module, input, output):
    bn2_cache['emb'] = output

hook = gnn_model.bn2.register_forward_hook(save_bn2_output)

with torch.no_grad():
    _ = gnn_model(x_numeric, x_cat, dummy_edge_index)

hook.remove()

gnn_emb = bn2_cache['emb'].detach().cpu().tolist()
gnn_emb = np.array(gnn_emb, dtype=np.float32)

# Build hybrid features
raw_numeric = test_df.loc[row_idx, TRAIN_NUMERIC_FEATURE_LIST].values.reshape(1, -1)
raw_cat = test_df.loc[row_idx, cat_cols].values.reshape(1, -1)
raw_features = np.hstack([raw_numeric, raw_cat])

hybrid_features = np.hstack([raw_features, gnn_emb])

dmatrix = xgb.DMatrix(hybrid_features)

final_prob = float(xgb_model.predict(dmatrix)[0])
# Apply threshold
is_fraud = final_prob >= threshold

# Display results
st.metric("Fraud Probability", f"{final_prob * 100:.2f}%")
st.write(f"**Threshold:** {threshold:.2f}")

if is_fraud:
    st.error("⚠️ This transaction is flagged as **FRAUD**.")
else:
    st.success("✅ This transaction is **NOT fraud**.")


# 4.4 4_About_Me.py

import streamlit as st

st.title("🌟 About Me")

st.write("""
I am a data scientist passionate about building intelligent systems that solve real-world problems responsibly.

My work focuses on:
- Machine learning and deep learning
- Cloud computing and big data systems
- Ethical and interpretable AI
- Scalable ML pipelines
- Fraud analytics
""")

st.write("""
I hold a Master’s in Data Science and a Bachelor’s in Software Engineering, and I enjoy 
exploring the intersection of technology, human behavior, and responsible innovation.

Outside of work, I love learning and sharing knowledge about data science, AI ethics, and the future of technology. I am also
an avid traveler and enjoy exploring new cultures and cuisines.
""")

# 4.5 Image Asset
img_dee.jpg

# 5. Model and data artifacts (filenames only)

models/
    edge_index.pt
    model_fraudsage2.pt
    test_encoded.parquet
    XGBoost_GNN.json

.vscode/models/
    model_fraudsage2.pt
    test_encoded.parquet
    XGBoost_GNN.json
