# Elbow Curve Visualization for K-Means Clustering

import matplotlib.pyplot as plt

def plot_elbow_curve(k_values, scores, title, save_path=None):
    plt.figure()
    plt.plot(list(k_values), scores, marker="o")
    plt.title(title)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


#-------------------------------------------------------------------------------------------------------------------------------
# PCA visualization
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

def plot_pca_clusters(x_scaled, labels, kmeans_model, save_path=None):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    cmap = ListedColormap(plt.get_cmap("Set2").colors[:3])              # Configure the colors for the plot

    plt.figure(figsize=(6, 4))
    plt.scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=labels,
        cmap=cmap,
        s=50,
    )
    plt.title("K-Means Clustering (2D PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    centroids_scaled = kmeans_model.cluster_centers_
    centroids_pca = pca.transform(centroids_scaled)
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=200,
        c="black",
        label="Centroids",
    )

    legend_labels = [
        mpatches.Patch(color=cmap(0), label="Cluster 0"),
        mpatches.Patch(color=cmap(1), label="Cluster 1"),
        mpatches.Patch(color=cmap(2), label="Cluster 2"),
    ]
    plt.legend(handles=legend_labels, title="Clusters", loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()