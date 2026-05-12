# 🕵️‍♂️ Fraud Analytics  
A graph‑enhanced and gradient‑boosted machine learning project for detecting fraudulent financial transactions using temporal validation, structured preprocessing, relational modeling, and rigorous evaluation.

---

## 📂 Project Overview  
This project builds, evaluates, and compares multiple fraud‑detection models using the IEEE‑CIS Fraud Detection dataset.  
The workflow includes temporal data splitting, extensive preprocessing, graph construction, GNN modeling, XGBoost baselines, and hybrid architectures.

Through empirical experimentation, the project demonstrates that **tabular XGBoost models outperform GNN‑only and hybrid models**, while GNNs still provide valuable relational insights.

---

## 🧹 Data Acquisition & Preparation  

### **Data Source**
- IEEE‑CIS Fraud Detection dataset (Kaggle)  
- 500k+ transactions across **Transaction** and **Identity** tables  
- Joined using `TransactionID`

### **Temporal Split**
- 50/50 time‑based split (train = earlier period, test = future period)  
- Prevents leakage and mimics real‑world deployment

### **Exploratory Data Analysis**
- Missingness patterns across identity features  
- Distribution of `TransactionAmt` and time deltas  
- High‑cardinality categorical exploration (emails, cards, devices)  
- Correlation analysis of masked V/C features  
- Structural patterns for graph construction (shared cards, devices, IPs)

### **Cleaning & Feature Engineering**
- Missing values encoded as explicit categories  
- Log‑transform applied to `TransactionAmt`  
- Cyclical encoding of `TransactionDT` (hour/day sin‑cos)  
- Known‑token reduction for emails, devices, and rare categories  
- Downcasting numeric dtypes for memory efficiency  
- All categorical features converted to `category` dtype  
- Consistent numeric encoding across train/test  

---

## 🧠 Baseline Model — Logistic Regression  
A simple baseline trained on cleaned tabular features.

**Performance:**  
- **AUC:** 0.87  
- Demonstrated strong predictive signal even without relational modeling

---

## 🔗 Graph Construction & GNN Models  

### **Graph Structure**
- Nodes = transactions  
- Edges = shared card identifiers (`card1`, `card1+card2`)  

### **GraphSAGE Experiments**

#### **1 Edge (card1)**
- **AUC:** 0.9302  
- High recall but high false positives  

#### **2 Edges (card1 + card2)**
- **AUC:** 0.9364  
- Higher precision  
- Higher recall  
- Fewer false positives  
- Strongest GNN variant  

#### **Ablation Study**
- Removing V‑features → ~8% performance drop  

#### **Supervised Contrastive Learning (SupCon)**
- Minimal AUC gain (+0.0001)  
- Reduced false positives but added instability  
- Not worth added complexity  

---

## ⚡ XGBoost — Tabular Baseline  
XGBoost delivered the strongest standalone performance.

**Results:**  
- **AUC:** 0.9566  
- **Precision:** 0.3352  
- **Recall:** 0.8352  
- False positives cut nearly in half vs. GNNs  
- Demonstrated that tabular patterns dominate predictive power in this dataset  

---

## 🧪 Hybrid Models (GNN + XGBoost)  

### **GNN Embeddings + XGBoost**
- **AUC:** 0.9524  
- Precision increased sharply  
- Recall dropped → more conservative model  

### **GNN + SupCon + XGBoost**
- **AUC:** 0.9508  
- Added complexity without meaningful benefit  

### **Threshold Optimization**
Optimal operating point identified at threshold **0.6450**:  
- **Precision:** 0.7121  
- **Recall:** 0.7002  

Balanced tradeoff for real‑world fraud operations.

---

## 🖥️ User Interface (Streamlit)  
The final Streamlit dashboard includes:  
- Transaction lookup by `TransactionID`  
- Fraud probability score from the hybrid model  
- Threshold slider for dynamic decisioning  
- Clean, user‑centric layout with real test‑set data  

---

## 📊 Tools & Technologies  
- **Python**  
- **Pandas, NumPy**  
- **Scikit‑Learn, XGBoost, Imbalanced‑Learn**  
- **PyTorch Geometric (GraphSAGE)**  
- **UMAP / t‑SNE**  
- **Matplotlib, Seaborn**  
- **Streamlit**  
- **Jupyter Notebook**

---

## ⭐ If you find this project interesting, feel free to star the repo!
