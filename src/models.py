# Computation of Elbow Method Scores for K-Means Clustering
from sklearn.cluster import KMeans

def compute_elbow_scores(X, n_init=100, random_state=200):

    # Calculates the WCSS for each k in range 1,11 (elbow method). To do this, we create a “score” list, in which we will store all WCSS values between 1 and 11. The function returns the score for each K.

    scores = []
    for k in range(1,11):
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state,
        )
        kmeans.fit(X)
        scores.append(kmeans.inertia_)
    return scores

