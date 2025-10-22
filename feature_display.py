import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your extracted features CSV
df = pd.read_csv("extracted_voice_features.csv")

# Create output directory
os.makedirs("figures", exist_ok=True)

# Remove non-numeric columns
numeric_df = df.drop(columns=["speaker", "file"])
labels = df["speaker"]

# 1. Boxplots for each feature
for col in numeric_df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="speaker", y=col)
    plt.title(f"Feature Distribution: {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/boxplot_{col}.png")
    plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(14, 12))
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png")
plt.close()

# 3. PCA visualization (2D)
X_scaled = StandardScaler().fit_transform(numeric_df)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10")
plt.title("PCA of Voice Features (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Speaker")
plt.tight_layout()
plt.savefig("figures/pca_2d.png")
plt.close()

# 4. Pairplot for selected features
subset_cols = [c for c in numeric_df.columns if "mean_mfcc_" in c][:3] + \
    ["pitch_mean", "rms_mean", "spec_centroid_mean"]
pairplot_df = df[subset_cols + ["speaker"]]

sns.pairplot(pairplot_df, hue="speaker", corner=True)
plt.suptitle("Pairplot of Selected Features by Speaker", y=1.02)
plt.savefig("figures/pairplot_selected_features.png")
plt.close()

print("All visualizations saved in the 'figures/' folder.")
