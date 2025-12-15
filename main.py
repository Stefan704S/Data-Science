#1) Importing the data and visualization

from src.data_loader import data_load
data = data_load("data/raw/data.xlsx")
print(data)


#2) Data visualization

print(data.describe())


#5)   Plot WCSS and choose k*

print("\n=== Elbow (full variables) ===")
    k_range = range(1, 11)
    score_h0 = compute_elbow_scores(x_full_scaled, k_range0)
    plot_elbow_curve(
        k_values=k_range,
        scores=score_h0,
        title="Elbow Method for Optimal k (full variables)",
        save_path="results/elbow_full.png",
    )
