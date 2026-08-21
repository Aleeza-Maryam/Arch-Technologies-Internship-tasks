# ============================================================
# TASK 3: CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# Make sure "Mall_Customers.csv" is uploaded in Colab/Jupyter
df = pd.read_csv("Mall_Customers.csv")

print("Dataset loaded successfully!")
print()


# ------------------------------------------------------------
# 3. BASIC DATA EXPLORATION
# ------------------------------------------------------------

print("First 5 rows:")
# NEW
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 4. REMOVE DUPLICATES
# ------------------------------------------------------------

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ------------------------------------------------------------
# 5. SELECT FEATURES FOR CLUSTERING
# ------------------------------------------------------------

# We will use:
# Age
# Annual Income
# Spending Score

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()


# ------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values in selected features:")
print(X.isnull().sum())

# Fill missing numerical values with median
X = X.fillna(X.median())

print("\nMissing values after cleaning:")
print(X.isnull().sum())


# ------------------------------------------------------------
# 7. VISUALIZE ORIGINAL FEATURES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    s=80
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()


# ------------------------------------------------------------
# 8. STANDARDIZE THE DATA
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully!")


# ------------------------------------------------------------
# 9. ELBOW METHOD
# ------------------------------------------------------------

inertia = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


# Plot Elbow Curve

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 10. APPLY K-MEANS
# ------------------------------------------------------------

# For this dataset, we use 5 clusters.
# You can change this value according to the Elbow Method.

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# ------------------------------------------------------------
# 11. ADD CLUSTER LABELS TO DATASET
# ------------------------------------------------------------

df["Cluster"] = clusters

print("\nCluster labels added successfully!")

print("\nFirst 10 rows with cluster labels:")
print(df.head(10))


# ------------------------------------------------------------
# 12. COUNT CUSTOMERS IN EACH CLUSTER
# ------------------------------------------------------------

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\nNumber of customers in each cluster:")
print(cluster_counts)


# ------------------------------------------------------------
# 13. SILHOUETTE SCORE
# ------------------------------------------------------------

silhouette = silhouette_score(
    X_scaled,
    clusters
)

print("\nSilhouette Score:")
print(round(silhouette, 4))


# ------------------------------------------------------------
# 14. CUSTOMER SEGMENT SUMMARY
# ------------------------------------------------------------

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCustomer Segment Summary:")
print(cluster_summary.round(2))


# ------------------------------------------------------------
# 15. ADD CUSTOMER COUNT TO SUMMARY
# ------------------------------------------------------------

cluster_summary["Customer Count"] = (
    df.groupby("Cluster").size()
)

print("\nComplete Cluster Summary:")
print(cluster_summary.round(2))


# ------------------------------------------------------------
# 16. VISUALIZE CUSTOMER SEGMENTS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")

plt.show()


# ------------------------------------------------------------
# 17. VISUALIZE AGE VS SPENDING SCORE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Age vs Spending Score by Customer Segment")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")

plt.show()


# ------------------------------------------------------------
# 18. VISUALIZE AGE VS ANNUAL INCOME
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Annual Income (k$)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Age vs Annual Income by Customer Segment")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.legend(title="Cluster")

plt.show()


# ------------------------------------------------------------
# 19. CREATE A SIMPLE DESCRIPTION OF EACH CLUSTER
# ------------------------------------------------------------

print("\n" + "="*60)
print("CUSTOMER SEGMENT CHARACTERISTICS")
print("="*60)

for cluster in sorted(df["Cluster"].unique()):

    age = cluster_summary.loc[cluster, "Age"]
    income = cluster_summary.loc[cluster, "Annual Income (k$)"]
    spending = cluster_summary.loc[
        cluster,
        "Spending Score (1-100)"
    ]

    count = int(
        cluster_summary.loc[cluster, "Customer Count"]
    )

    print(f"\nCluster {cluster}")
    print("-" * 40)

    print(f"Number of Customers: {count}")
    print(f"Average Age: {age:.2f}")
    print(f"Average Annual Income: {income:.2f} k$")
    print(f"Average Spending Score: {spending:.2f}")

    # Automatically describe the cluster

    if income >= df["Annual Income (k$)"].mean() and spending >= df["Spending Score (1-100)"].mean():

        print("Description: High-income, high-spending customers.")

    elif income >= df["Annual Income (k$)"].mean() and spending < df["Spending Score (1-100)"].mean():

        print("Description: High-income, low-spending customers.")

    elif income < df["Annual Income (k$)"].mean() and spending >= df["Spending Score (1-100)"].mean():

        print("Description: Low-income, high-spending customers.")

    else:

        print("Description: Low-income, low-spending customers.")


# ------------------------------------------------------------
# 20. SAVE FINAL DATASET
# ------------------------------------------------------------

df.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nFinal segmented dataset saved as:")
print("customer_segments.csv")


# ------------------------------------------------------------
# 21. FINAL PROJECT SUMMARY
# ------------------------------------------------------------

print("\n" + "="*60)
print("PROJECT COMPLETED")
print("="*60)

print("""
Customer segmentation was successfully performed using
K-Means clustering.

Steps performed:
1. Loaded the customer dataset
2. Explored the data
3. Checked missing values and duplicates
4. Selected relevant features
5. Standardized the features
6. Used the Elbow Method
7. Applied K-Means clustering
8. Evaluated clustering using Silhouette Score
9. Visualized customer segments
10. Analyzed characteristics of each segment
11. Saved the final segmented dataset
""")