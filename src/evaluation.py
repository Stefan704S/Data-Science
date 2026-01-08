'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Evaluation & Visualization
--------------------------------------------------------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Import
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Librairies

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Elbow Curve
--------------------------------------------------------------------------------------------------------------------------------------
'''

#This function visualizes the Within-Cluster Sum of Squares (WCSS) across different values of k (number of clusters). The curve helps identify the optimal number of clusters by locating the "elbow point," where adding more clusters no longer significantly reduces  WCSS. 


def plot_elbow_curve(k_values, scores, title, save_path=None):
    plt.figure()
    plt.plot(list(k_values), scores, marker="o")                    # Configure the values for the plot : k-vlaues = x, scores = y
    plt.title(title)                                                
    plt.xlabel("Number of Clusters")                                
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        PCA Visualization
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Construction of a function that allows clusters to be visualized graphically. To do this, we reduce to two dimensions for 2D visualization. Implementation of all graph parameters. 

def plot_pca_clusters(x_scaled, labels, kmeans_model, save_path=None):

    pca = PCA(n_components=2)                                      
    x_pca = pca.fit_transform(x_scaled)                            # Reducing the variables dimensions to 2

    cmap = ListedColormap(plt.get_cmap("Set2").colors[:3])         # Configure the colors for the plot

    plt.figure(figsize=(6, 4))
    plt.scatter(
        x_pca[:, 0],
        x_pca[:, 1],
        c=labels,
        cmap=cmap,
        s=50)
    
    plt.title("K-Means Clustering (2D PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    centroids_scaled = kmeans_model.cluster_centers_               # Calculate the coordinates of the centroids
    centroids_pca = pca.transform(centroids_scaled)                # Reducing the centroids to 2 dimensions
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=200,
        c="black",
        label="Centroids")

    legend_labels = [
        mpatches.Patch(color=cmap(0), label="Cluster 0"),
        mpatches.Patch(color=cmap(1), label="Cluster 1"),
        mpatches.Patch(color=cmap(2), label="Cluster 2")]
    plt.legend(handles=legend_labels, title="Clusters", loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                        Build the different macro regimes over time
--------------------------------------------------------------------------------------------------------------------------------------
'''
# We want to check the distribution of clusters over time, so we will set up a timeline on which we will represent each cluster observation. The goal is to see if the observations follow each other to form macro regimes.

def time(df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = sns.color_palette("Set2", n_colors=df["Cluster_1"].nunique())

    sns.scatterplot(
        data=df,
        x="Dates",
        y="Cluster_1",
        hue="Cluster_1",
        palette=palette,
        s=24,
        alpha=0.7,
        edgecolor="none",
        ax=ax,
        legend="full")

    ax.set_yticks(sorted(df["Cluster_1"].unique()))
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_xlabel("Dates", fontsize=12)
    ax.set_title(
        "The different macro regimes according to clusters over time",
        fontsize=14,
        weight="bold",
        y=1.1)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.margins(x=0.01, y=0.2)

    ax.legend(
        title="Cluster",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.08),
        frameon=True,
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Regression Plot
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Construction of a function that displays the regressions for each regime. To do this, we create a for loop in which we configure a model that we integrate for each cluster. This gives us an image with three regressions on the same line.

def regression(data, models, Y, X, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for i, cluster in enumerate(sorted(models.keys())):
        model = models[cluster]

        y_pred = model.fittedvalues
        y_var = model.model.endog

        ax = axs[i]
        ax.scatter(
            y_pred,
            y_var,
            alpha=0.6,
            color="#1f77b4",
            edgecolor="black")

        ax.plot(
            [y_pred.min(), y_pred.max()],
            [y_pred.min(), y_pred.max()],
            color="red",
            linestyle="--")

        ax.set_title(f"Cluster {cluster}", fontsize=13, weight="bold")
        ax.set_xlabel("Predicted Inflation")
        if i == 0:
            ax.set_ylabel("Observed Inflation")
        ax.grid(True, linestyle="--", alpha=0.8)

    plt.suptitle(
        "Multiple regression: observed vs predicted inflation by cluster",
        fontsize=16,
        weight="bold",
        y=1)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                Heteroscedasticity Plot
--------------------------------------------------------------------------------------------------------------------------------------
'''
# As with regressions, we want to observe how the residuals are distributed in the regressions. We also apply a for loop to iterate through each cluster. Single-line visualization.

def hetero_plot(data, models, Y, X, save_path=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for i, cluster in enumerate(sorted(models.keys())):
        model = models[cluster]
        y_pred = model.fittedvalues
        residuals = model.resid
        std_resid = np.std(residuals)

        ax = axs[i]
        ax.scatter(
            y_pred,
            residuals,
            alpha=0.6,
            color="#1f77b4",
            edgecolor="black")

        ax.axhline(0, color="red", linestyle="--", linewidth=1)

        ax.plot(
            [y_pred.min(), y_pred.max()],
            [2 * std_resid, 2 * std_resid],
            color="gray",
            linestyle="--")
        
        ax.plot(
            [y_pred.min(), y_pred.max()],
            [-2 * std_resid, -2 * std_resid],
            color="gray",
            linestyle="--")

        sns.regplot(
            x=y_pred,
            y=residuals,
            lowess=True,
            ax=ax,
            scatter=False,
            color="green",
            line_kws={"linewidth": 2})

        ax.set_title(f"Cluster {cluster}", fontsize=13, weight="bold")
        ax.set_xlabel("Fitted values")
        if i == 0:
            ax.set_ylabel("Residuals")
        ax.grid(True, linestyle="--", alpha=0.8)

    plt.suptitle(
        "Residuals vs Fitted values with ±2σ cones and LOWESS by cluster",
        fontsize=16,
        weight="bold",
        y=1.05)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
