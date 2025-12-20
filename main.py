'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                            Main.py
--------------------------------------------------------------------------------------------------------------------------------------
'''


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                            Import
--------------------------------------------------------------------------------------------------------------------------------------
'''

# Libraries

from src.models import compute_elbow_scores
from src.models import kmeans
from src.models import anova
from src.models import silhouette

# Local imports

from src.data_loader import data_load, standardize


from src.models import (
    compute_elbow_scores,
    kmeans,
    anova,
    silhouette,
    build_clustered_data,
    vif,
    ols,
    robust)


from src.evaluation import (
    plot_elbow_curve,
    plot_pca_clusters,
    time,
    regression,
    hetero_plot)


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Data import
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Importing the data and visualization.

data = data_load("data/raw/data.xlsx")
print(data)

'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Clustering
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Results for the hypothesis 0
# Plot WCSS and choose k*

# In this part of the code, we select the variables we want to analyze for KMeans. To do this, we standardize them, calculate the Elbow score, and implement it in the graph.

vars_h0 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF", "GDP", "SMI"]
x_full_scaled = standardize(data[vars_h0])
print("\n=== Elbow (full variables) ===")
k_range = range(1, 11)

score_h0 = compute_elbow_scores(x_full_scaled)

plot_elbow_curve(
    k_values=k_range,
    scores=score_h0,
    title="Elbow Method for Optimal k (full variables)",
    save_path="results/plot/elbow_full.png"
)


# Run Anova and Silhouette test
# K-means with k = 3 and all the variables

# In this part of the code, we test clustering. We want to know if the variables are useful in determining the cluster and the quality of the cluster through its silhouette test. We therefore refer to “elbow_full” to determine the number n_clusters in kmeans.

labels_0, kmeans_0 = kmeans(x_full_scaled, n_clusters=3)
data["Cluster_0"] = labels_0

# Anova test
variables_0 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF", "GDP", "SMI"]
anova_h0 = anova(data, cluster_col="Cluster_0", variables=variables_0)
print(anova_h0)
anova_h0.to_csv("results/numeric/Anova_0.csv", index=False)

# Silhouette test
sil_0 = silhouette(x_full_scaled, labels_0)
print(sil_0)
sil_0.to_csv("results/numeric/Sil_h0.csv", index=False)


#-------------------------------------------------------------------------------------------------------------------------------
# Results for the hypothesis 1
# Plot WCSS and choose k*

# Same procedure as in hypothesis 0, only the variables change.

vars_h1 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF"]
x_new_scaled = standardize(data[vars_h1])
print("\n=== Elbow (full variables) ===")
k_range = range(1, 11)

score_h1 = compute_elbow_scores(x_new_scaled)

plot_elbow_curve(
    k_values=k_range,
    scores=score_h1,
    title="Elbow Method for Optimal k (full variables)",
    save_path="results/plot/elbow_new.png",
)


# Run Anova and Silhouette test
# K-means with k = 3 and all the variables
labels_1, kmeans_1 = kmeans(x_new_scaled, n_clusters=3)
data["Cluster_1"] = labels_1

# Anova test
variables_1 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF"]
anova_1 = anova(data, cluster_col="Cluster_1", variables=variables_1)
print(anova_1)
anova_1.to_csv("results/numeric/Anova_h1.csv", index=False)

# Silhouette test
sil_1 = silhouette(x_new_scaled, labels_1)
print(sil_1)
sil_1.to_csv("results/numeric/Sil_h1.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------
# Analysis of the clustered dataset
# Saving the relevant infos

data_clustered = build_clustered_data(data)
print(data_clustered.head())

cluster_counts = data_clustered["Cluster_1"].value_counts().sort_index()
print(cluster_counts)
cluster_counts.to_csv("results/numeric/Cluster_counts.csv", index=False)

variables_1 = ["3Mth", "10Yd", "Inf", "Unmp", "CHF"]
cluster_means = data_clustered.groupby("Cluster_1")[variables_1].mean()
print(cluster_means)
cluster_means.to_csv("results/numeric/Cluster_means.csv", index=False)


#-------------------------------------------------------------------------------------------------------------------------------
# Ploting PCA with the function in th evaluation 

plot_pca_clusters( x_scaled=x_new_scaled, labels=labels_1, kmeans_model=kmeans_1, save_path="results/pca_clusters_macro.png",)


#-------------------------------------------------------------------------------------------------------------------------------
# Ploting the cluster over time

time(df=data_clustered, save_path="results/plot/clusters_over_time.png")


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Regressions (OLS)
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Call the fucntion to check the vif for the remaining variables
# We suspect a problem of multicollinearity in the data, particularly between 3mth and 10yd, so we test once with all the data and a second time removing 10yd.

# Hypothesis 0
variables_vif_0 = ["3Mth", "10Yd", "Unmp", "CHF"]
vif_table_0= vif(data_clustered, variables_vif_0)
print("\nVif test :")
print(vif_table_0.round(2))
vif_table_0.to_csv("results/numeric/Vif_h0.csv", index=False)


# Hypothesis 1
variables_vif_1 = ["3Mth", "Unmp", "CHF"]
vif_table_1 = vif(data_clustered, variables_vif_1)
print("\nVif test :")
print(vif_table_1.round(2))
vif_table_1.to_csv("results/numeric/Vif_h1.csv", index=False)


#-------------------------------------------------------------------------------------------------------------------------------
# Call the function to do the regression
# We perform regression with 3mth, Unmp, and CHF, removing 10yd due to multicollinearity.

x = ['3Mth', 'Unmp', 'CHF']
y = 'Inf'

ols_table, models = ols(df=data_clustered, Y=y, X=x)
print(ols_table)
ols_table.to_csv("results/numeric/OLS_table.csv", index=False)


#-------------------------------------------------------------------------------------------------------------------------------
# Call the function for the robustness test
# We use a robustness test with the aim of potentially improving p-values, due to the presence of heteroscedasticity. We use the “HC3” test because we have a small cluster of 33.

rob_table = robust(data_clustered, Y=y, X=x, cov_type="HC3")
print(rob_table.round(4))
rob_table.to_csv("results/numeric/Robustness_test.csv", index=False)


#-------------------------------------------------------------------------------------------------------------------------------
# Call the function to plot the regression
# We plot regression with 3mth, Unmp, and CHF.

regression(data=data_clustered, models=models, Y=y, X=x, save_path="results/plot/regression")


#-------------------------------------------------------------------------------------------------------------------------------------
# Call the function to plot the heteroscedasticity

hetero_plot(data=data_clustered, models=models, Y=y, X=x, save_path="results/plot/Hetero_plot")
