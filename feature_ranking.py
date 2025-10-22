import pandas as pd
import numpy as np

# Load your extracted features CSV
df = pd.read_csv("extracted_voice_features.csv")

# Filter out non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
label_col = "speaker"

# Group by speaker and compute stats
grouped = df.groupby(label_col)[numeric_cols]

# Inter-speaker variance (how much means vary between users)
inter_speaker_variance = grouped.mean().var()

# Intra-speaker variance (how much features vary within each speaker)
intra_speaker_variance = grouped.var().mean()

# Discriminability score: higher = better for distinguishing speakers
discriminability = inter_speaker_variance / (intra_speaker_variance + 1e-6)

# Rank features
ranking = discriminability.sort_values(ascending=False)
ranking.name = "discriminability_score"

# Save ranking to CSV
ranking.to_csv("figures/feature_discriminability_ranking.csv")

# Print top 15
print("Top 15 most informative features:")
print(ranking.head(15))
