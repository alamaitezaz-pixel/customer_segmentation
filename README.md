# Customer Segmentation with K-Means

An unsupervised machine learning project that groups wholesale customers by their spending patterns. Built with Python and scikit-learn in Google Colab.

## Dataset

The UCI Wholesale Customers dataset contains 440 customers and eight columns. Six spending categories are used for clustering:

- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

`Channel` identifies the customer's business type, and `Region` identifies their location category. These columns are excluded from model training; Channel is used afterward to help interpret the clusters. There is no target column because this is an unsupervised learning project.

## Workflow

1. Inspect the dataset and check for missing values.
2. Explore spending distributions with histograms.
3. Apply `log1p` to reduce skewness and `StandardScaler` to standardize the six spending features.
4. Compare K-Means models with 2–10 clusters using silhouette scores.
5. Examine cluster sizes, spending medians, quartiles, and business-type distributions.
6. Check the three-cluster solution across different random seeds using silhouette scores and Adjusted Rand Index (ARI).
7. Fit the final model and save it with its scaler, feature order, and transformation metadata using Joblib.
8. Reload the saved bundle and verify that its predictions match the original assignments.

## Results

The final model uses **three clusters**, with a silhouette score of approximately **0.259**.

| Cluster | Customers | Spending pattern |
| --- | ---: | --- |
| 0 | 80 | Grocery and household emphasis, with lower fresh and frozen spending |
| 1 | 147 | Broad spending across product categories |
| 2 | 213 | Fresh and frozen emphasis, with lower milk, grocery, and detergent spending |

The two-cluster candidate had a higher silhouette score. Three clusters were selected to provide more detailed customer profiles. The final score indicates overlapping groups, so these segments should be treated as exploratory descriptions.

Across the tested random seeds, ARI values were approximately **0.94–0.96** compared with the original three-cluster solution, indicating similar assignments. Reloading the saved final model reproduced its original assignments.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, Joblib, and Google Colab.

## How to Run

1. Open the project notebook in Google Colab.
2. Run the cells from top to bottom. The notebook downloads the dataset automatically.
3. Mount Google Drive when prompted to save the model bundle as `customer_segmentation.joblib`.
4. Run the reload section to verify the saved model.

For new customers, use the saved feature order, apply `log1p`, transform with the saved scaler, and then call the saved model's `predict` method.

## Potential Use

These segments could help a wholesaler explore product bundles and targeted offers. Their business impact has not been tested. Cluster numbers are labels and may change when the model is retrained.

## Dataset Source

UCI Machine Learning Repository — Wholesale Customers dataset.
