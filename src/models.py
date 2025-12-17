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


#------------------------------------------------------------------------
# Adding for each observations the corresponding clusters

def build_clustered_data(data: pd.DataFrame) -> pd.DataFrame:

    # This function allows you to create, from the initial database, a new dataset in which the cluster corresponding to each observation is added.

    selected_vars = ["Dates", "Cluster_1", "3Mth", "10Yd", "Inf", "Unmp", "CHF"]
    df = data[selected_vars].copy()
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates")
    return df


#-----------------------------------------------------------------------------------------------------------------------------
# Testing the multicolinearity between the variables 

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(data_clustered: pd.DataFrame, x_vars):
    
    # Calculates the VIF for the specified variables. Returns a DataFrame with Variable and VIF.
    
    X_vif = sm.add_constant(data_clustered[x_vars])
    vif_table = pd.DataFrame(
        {
            "Variable": X_vif.columns,
            "VIF": [
                variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])
            ],
        }
    )
    return vif_table


#-------------------------------------------------------------------------------------------------------------------------------
# Computing the Regression 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

def ols(df, Y, X):
    
    # This function groups observations by cluster and estimates an OLS linear regression for each cluster. It returns the coefficients, p-values, R^2, and a test for heteroscedasticity.

    summary_table = []

    for c in sorted(df["Cluster_1"].unique()):
        subset = df[df["Cluster_1"] == c]

        x_var= sm.add_constant(subset[X])
        y_var = subset[Y]

        model = sm.OLS(y_var, x_var).fit()
        bp_test = het_breuschpagan(model.resid, model.model.exog)

        summary_table.append(
            {
                "Intercept": model.params["const"],
                "Beta (3Mth)": model.params.get("3Mth", np.nan),
                "Pval (3Mth)": model.pvalues.get("3Mth", np.nan),
                "Beta (Unmp)": model.params.get("Unmp", np.nan),
                "Pval (Unmp)": model.pvalues.get("Unmp", np.nan),
                "Beta (CHF)": model.params.get("CHF", np.nan),
                "Pval (CHF)": model.pvalues.get("CHF", np.nan),
                "R²": model.rsquared,
                "Adj R²": model.rsquared_adj,
                "BP p-value": bp_test[1],
                "N obs": int(model.nobs),
            }
        )

    summary_df = pd.DataFrame(summary_table)
    return summary_df

