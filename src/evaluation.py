# Elbow Curve Visualization for K-Means Clustering

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