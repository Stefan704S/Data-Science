'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Models & Statistical Tests
--------------------------------------------------------------------------------------------------------------------------------------
'''

'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Import
--------------------------------------------------------------------------------------------------------------------------------------
'''

# Librairies

from sklearn.cluster import KMeans
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Elbow Method
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Calculates the WCSS for each k in range 1,11 (elbow method). To do this, we create a “score” list, in which we will store all WCSS values between 1 and 11. The function returns the score for each K.

def compute_elbow_scores(X, n_init=100, random_state=200):
    
    scores = []

    for k in range(1, 11):
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=random_state)
        
        kmeans.fit(X)
        scores.append(kmeans.inertia_)

    return scores


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        K-Means Clustering
--------------------------------------------------------------------------------------------------------------------------------------
'''
# This function applies the K-means algorithm to the X data in order to partition the observations into n_clusters groups. It returns the cluster label associated with each observation as well as the trained KMeans model.

def kmeans(X, n_clusters=3, n_init=100, random_state=200):
  
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state)
    
    labels = kmeans.fit_predict(X)
    return labels, kmeans


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        ANOVA Test
--------------------------------------------------------------------------------------------------------------------------------------
'''
 # This function performs a one-way ANOVA for each specified variable, to test whether the means differ significantly between clusters. It displays the p-value associated with each test.

def anova(data: pd.DataFrame, cluster_col: str, variables):
   
    results = []
   
    for var in variables:
        groups = [
            data[data[cluster_col] == k][var]
            for k in sorted(data[cluster_col].unique())]
        stat, p = f_oneway(*groups)
        results.append({
            "Variable": var,
            "F-statistic": stat,
            "p-value": p })
        print(f"{var}: p-value = {p:.4f}")

    return pd.DataFrame(results)


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Silhouette Test
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Computing the Silhouette test

def silhouette(X, labels):

    result_sil = silhouette_score(X, labels)
    return pd.DataFrame({"Silhouette score": [result_sil]})


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Build Clustered Dataset
--------------------------------------------------------------------------------------------------------------------------------------
'''
# This function allows you to create, from the initial database, a new dataset in which the cluster corresponding to each observation is added.

def build_clustered_data(data: pd.DataFrame):
   
    selected_vars = ["Dates", "Cluster_1", "3Mth", "10Yd", "Inf", "Unmp", "CHF"]
    df = data[selected_vars].copy()
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates")
    return df


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Multicollinearity (VIF)
--------------------------------------------------------------------------------------------------------------------------------------
'''
# Calculates the VIF for the specified variables. Returns a DataFrame with Variable and VIF.

def vif(data_clustered: pd.DataFrame, x_vars):

    X_vif = sm.add_constant(data_clustered[x_vars])
    vif_table = pd.DataFrame(
        {"Variable": X_vif.columns,
         "VIF": [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]})
    
    return vif_table


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        OLS Regression
--------------------------------------------------------------------------------------------------------------------------------------
'''
# This function groups observations by cluster and estimates an OLS linear regression for each cluster. It returns the coefficients, p-values, R^2, and a test for heteroscedasticity.

def ols(df, Y, X):

    models = {}
    summary_table = []

    for c in sorted(df["Cluster_1"].unique()):
        
        subset = df[df["Cluster_1"] == c]
        
        x_var = sm.add_constant(subset[X])
        y_var = subset[Y]
        
        model = sm.OLS(y_var, x_var).fit()
        models[c] = model
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
                "BP p-value": bp_test[1],
                "N obs": int(model.nobs),})
        
    summary_df = pd.DataFrame(summary_table)
    return summary_df, models


'''
--------------------------------------------------------------------------------------------------------------------------------------
                                                        Robustness Test
--------------------------------------------------------------------------------------------------------------------------------------
'''
# This function applies the robust test for each cluster to correct the initial p-values. To do this, we use two loops: the first performing regression on the clusters and the second performing robustness testing for each cluster.

def robust(df, Y, X, cov_type):

    robust_models = []

    for c in sorted(df["Cluster_1"].dropna().unique()):
        subset = df[df["Cluster_1"] == c].dropna(subset=[Y] + list(X))
        X_mat = sm.add_constant(subset[X], has_constant="add")
        y_vec = subset[Y]
        model = sm.OLS(y_vec, X_mat).fit()
        res = model.get_robustcov_results(cov_type=cov_type)
        names = list(model.params.index)
        row = {"Cluster": c}

        for i, name in enumerate(names):
            row[f"Robust Pval ({name})"] = float(res.pvalues[i]) if i < len(res.pvalues) else np.nan
        robust_models.append(row)

    return pd.DataFrame(robust_models).sort_values("Cluster").reset_index(drop=True)
