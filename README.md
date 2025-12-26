# **Customer Financial Risk Segmentation: A Multi-Model Clustering & PCA Approach**

## **1. Executive Summary**

This project presents a robust analytical framework designed to identify distinct customer personas and assess credit default risk using **Unsupervised Learning**. By integrating **Hierarchical Clustering**, **K-Means**, and **Principal Component Analysis (PCA)**, the analysis transforms raw financial data into actionable segments. The core objective is to isolate "High Default Risk" profiles, enabling data-driven strategies for risk mitigation and compliance, critical for fintech environments .

---

## **2. Dataset Architecture**

The model processes high-volume financial datasets containing the following variables:

* **Income Metrics:** Per Capita Income, Yearly Income.
* **Credit Health:** Credit Score, Number of Credit Cards.
* **Liability & Usage:** Total Debt, Transaction Amount.

---

## **3. Methodology & Core Code Implementation**

### **3.1 Data Preprocessing & Scaling**

To ensure variables with different scales (e.g., Income vs. Credit Score) don't bias the model, the data is normalized.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(raw_data)

```

### **3.2 Dimensionality Reduction via PCA**

**Logic:** Financial variables are often highly correlated (e.g., higher income usually correlates with higher credit limits). PCA is applied to reduce noise and multicollinearity while retaining maximum variance.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

```

* **Significance:** PCA improved cluster stability and enabled 2D/3D visualization of latent financial behaviors.

### **3.3 Hierarchical Clustering (Risk Identification)**

**Logic:** This agglomerative approach builds a hierarchy of clusters, visualized via a Dendrogram to determine the optimal number of segments based on Euclidean distance.

```python
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(pca_data, method='ward')
dendrogram(Z)

```

* **Impact:** This method proved strongest for **Risk Separation**, successfully isolating a cluster with "High Debt relative to Income" and the "Lowest Credit Score."

### **3.4 K-Means Clustering (Operational Segmentation)**

**Logic:** K-Means partitions the data into  non-overlapping subgroups. Here,  was selected as the optimal number of clusters.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(pca_data)

```

* **Impact:** Best for **Income-based Segmentation**, providing a clear view of enterprise performance across customer tiers.



---

## **4. Strategic Findings: The Three Customer Personas**

| Persona | Financial Profile | Credit Health | Default Risk | Recommended Strategy |
| --- | --- | --- | --- | --- |
| **The Cautious Segment** | Low Income / Low Debt | Good Score | **Low** | Basic Digital Banking |
| **The Premium Segment** | High Income / High Debt | **Strongest Score** | **Medium** | Credit Expansion / Premium Products |
| **The At-Risk Segment** | Mid Income / Very High Debt | **Lowest Score** | **High** | **Targeted Risk Mitigation** |

---

## **5. Technical Rigor & Quality Standards**

To ensure the model meets the high-quality data standards required in retail and fintech analytics, the following checks were performed:

* **Cluster Stability:** PCA-enhanced K-Means showed significantly lower noise and higher silhouette scores compared to raw K-Means.
* **Statistical Integrity:** Variables like Debt-to-Income and Credit Score showed high significance in defining cluster centroids.
* **Data Integrity:** Improved model reliability through outlier capping and encoding, ensuring stable and interpretable predictors.



---

## **6. Business Impact & Conclusion**

* **Fraud Detection:** The segmentation framework assists in detecting fraud patterns by identifying anomalies in high-usage, low-credit-score clusters.


* **Efficiency:** Automating these risk-profiling workflows increases analytical efficiency, similar to the 27% gain achieved in previous financial reporting projects.


* **Final Outcome:** Provides a data-driven intelligence layer for executive decision-making, transforming complex datasets into clear, actionable risk narratives.
