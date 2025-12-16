# 1) Importing the data and visualization

from src.data_loader import data_load

data = data_load("data/raw/data.xlsx")
print(data)


# ----------------------------------------------------------------------------------------------
# 2) Data visualization

print(data.describe())


# ----------------------------------------------------------------------------------------------
# 5)   Plot WCSS and choose k*

from src.models import compute_elbow_scores
from src.evaluation import plot_elbow_curve
from src.data_loader import standardize

x_full_scaled = standardize(data)

print("\n=== Elbow (full variables) ===")
k_range = range(1, 11)

score_h0 = compute_elbow_scores(x_full_scaled)

plot_elbow_curve(
    k_values=k_range,
    scores=score_h0,
    title="Elbow Method for Optimal k (full variables)",
    save_path="results/elbow_full.png",
)


#-------------------------------------------------------------------------------------------------------------------------------
#Results for the hypothesis 0

from src.models import kmeans
from src.models import anova
from src.models import silhouette

# K-means with k = 3 and all the variables
labels_0, kmeans_0 = kmeans(x_full_scaled, n_clusters=3)
data["Cluster_0"] = labels_0

variables_0 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF", "GDP", "SMI"]
anova(data, cluster_col="Cluster_0", variables=variables_0)

sil_0 = silhouette(x_full_scaled, labels_0)
print(sil_0)
