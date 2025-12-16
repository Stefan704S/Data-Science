# 1) Importing the data and visualization

from src.data_loader import data_load

data = data_load("data/raw/data.xlsx")
print(data)


# ----------------------------------------------------------------------------------------------
# 2) Data visualization

print(data.describe())


# ----------------------------------------------------------------------------------------------
#Results for the hypothesis 0

from src.models import compute_elbow_scores
from src.evaluation import plot_elbow_curve
from src.data_loader import standardize
from src.models import kmeans
from src.models import anova
from src.models import silhouette


# 5)   Plot WCSS and choose k*

vars_h0 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF", "GDP", "SMI"]
x_full_scaled = standardize(data[vars_h0])
print("\n=== Elbow (full variables) ===")
k_range = range(1, 11)

score_h0 = compute_elbow_scores(x_full_scaled)

plot_elbow_curve(
    k_values=k_range,
    scores=score_h0,
    title="Elbow Method for Optimal k (full variables)",
    save_path="results/elbow_full.png",
)

# 6 Run Anova and Silhouette test

# K-means with k = 3 and all the variables
labels_0, kmeans_0 = kmeans(x_full_scaled, n_clusters=3)
data["Cluster_0"] = labels_0

# Anova test
variables_0 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF", "GDP", "SMI"]
anova(data, cluster_col="Cluster_0", variables=variables_0)

# Silhouette test
sil_0 = silhouette(x_full_scaled, labels_0)
print(sil_0)


#-------------------------------------------------------------------------------------------------------------------------------
#Results for the hypothesis 1

from src.models import compute_elbow_scores
from src.evaluation import plot_elbow_curve
from src.data_loader import standardize
from src.models import kmeans
from src.models import anova
from src.models import silhouette

# 7)   Plot WCSS and choose k*

vars_h1 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF"]
x_new_scaled = standardize(data[vars_h1])
print("\n=== Elbow (full variables) ===")
k_range = range(1, 11)

score_h1 = compute_elbow_scores(x_new_scaled)

plot_elbow_curve(
    k_values=k_range,
    scores=score_h1,
    title="Elbow Method for Optimal k (full variables)",
    save_path="results/elbow_new.png",
)

# 8 Run Anova and Silhouette test

# K-means with k = 3 and all the variables
labels_1, kmeans_1 = kmeans(x_new_scaled, n_clusters=3)
data["Cluster_1"] = labels_1

# Anova test
variables_1 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF"]
anova(data, cluster_col="Cluster_1", variables=variables_1)

# Silhouette test
sil_1 = silhouette(x_new_scaled, labels_1)
print(sil_1)


#------------------------------------------------------------------------------------------------------------------------------
# Analysis of the clustered dataset

from src.models import build_clustered_data

data_clustered = build_clustered_data(data)
print(data_clustered.head())


cluster_counts = data_clustered["Cluster_1"].value_counts().sort_index()
print(cluster_counts)


variables_1 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF"]
cluster_means = data_clustered.groupby("Cluster_1")[variables_1].mean()
print(cluster_means)


#-------------------------------------------------------------------------------------------------------------------------------
# Ploting PCA with the function in th evaluation 

from src.evaluation import plot_pca_clusters

plot_pca_clusters(
        x_scaled=x_new_scaled,
        labels=labels_1,
        kmeans_model=kmeans_1,
        save_path="results/pca_clusters_macro.png",)


#-------------------------------------------------------------------------------------------------------------------------------
# Ploting the cluster over time

from src.evaluation import time

time(df=data_clustered, save_path="results/clusters_over_time.png")


#-------------------------------------------------------------------------------------------------------------------------------
# Call the fucntion to check the vif for the remaining variables

from src.models import vif

# Hypothesis 0

variables_vif_0 = ["3Mth", "10Yd", "Unmp", "CHF"]
vif_table = vif(data_clustered, variables_vif_0)
print("\nVif test :")
print(vif_table.round(2))

# Hypothesis 1

variables_vif_1 = ["3Mth", "Unmp", "CHF"]
vif_table = vif(data_clustered, variables_vif_1)
print("\nVif test :")
print(vif_table.round(2))