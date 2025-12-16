# Computation of Elbow Method Scores for K-Means Clustering

from sklearn.cluster import KMeans

def compute_elbow_scores(X, n_init=100, random_state=200):


    # Calculates the WCSS for each k in range 1,11 (elbow method). To do this, we create a “score” list, in which we will store all WCSS values between 1 and 11. The function returns the score for each K.

    scores = []
    for k in range(1, 11):
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )
        kmeans.fit(X)
        scores.append(kmeans.inertia_)
    return scores


#-------------------------------------------------------------------------------------------------------------------------------
# Computating the PCA with K = 3 according to the graphics

def kmeans(X, n_clusters=3, n_init=100, random_state=200):
  
    # This function applies the K-means algorithm to the X data in order to partition the observations into n_clusters groups. It returns the cluster label associated with each observation as well as the trained KMeans model.

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    return labels, kmeans


#-------------------------------------------------------------------------------------------------------------------------------
# Computing the Anova test to test the varialbles
import pandas as pd
from scipy.stats import f_oneway


def anova(data : pd.DataFrame, cluster_col: str, variables):
  
# This function performs a one-way ANOVA for each specified variable, to test whether the means differ significantly between clusters. It displays the p-value associated with each test.

    for var in variables:
        groups = [
            data[data[cluster_col] == k][var]
            for k in sorted(data[cluster_col].unique())
        ]
        stat, p = f_oneway(*groups)
        print(f"{var}: p-value = {p:.4f}")


#-------------------------------------------------------------------------------------------------------------------------------
# Computing the silhouette test
from sklearn.metrics import silhouette_score

def silhouette(X, labels):
    return silhouette_score(X, labels)
