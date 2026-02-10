# PCA plot : Run separately for each slide and also once for combined without filtering slide

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Paths to input folder and results
BASE_FOLDER = Path("S:/266_and_267/PCA")
GLYCAN_CSV = BASE_FOLDER / "glycan_df.csv"
PEPTIDE_CSV = BASE_FOLDER / "peptide_df.csv"

PCA_OUTPUT = BASE_FOLDER / "PCA_Plots_both"
PCA_OUTPUT.mkdir(exist_ok=True, parents=True)

folders = {
    "glycan_tissue": PCA_OUTPUT / "Glycan_Tissue",
    "glycan_fixation": PCA_OUTPUT / "Glycan_Fixation",
    "peptide_tissue": PCA_OUTPUT / "Peptide_Tissue",
    "peptide_fixation": PCA_OUTPUT / "Peptide_Fixation"
}

for f in folders.values():
    f.mkdir(exist_ok=True)

# Load datasets
df_gly = pd.read_csv(GLYCAN_CSV) # load glycan
df_pep = pd.read_csv(PEPTIDE_CSV) # load peptide

# Remove matrix spots
df_gly = df_gly[~df_gly["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
df_pep = df_pep[~df_pep["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
#df_gly = df_gly[df_gly["Experiment"] == 267]
#df_pep = df_pep[df_pep["Experiment"] == 267]  # used for filtering slides , when -#- there is no filtering base on slide number and both slides will be used

# Color maps
## for tissue types
tissue_colors = {
    "brain": "#6699FF",
    "kidney": "#A9A9A9",
    "liver": "#B266FF",
    "lung": "#66CC66",
    "muscle": "#FF6F66",
    "pancreas": "#A0522D",
    "spleen": "#FFEB3B"
}
#for fixation groups
fixation_colors = {
    "3H": "#FF00FF",
    "6H": "#FFFF00",
    "12H": "#0000FF",
    "24H": "#FFA500",
    "60H": "#00FF00"
}

#pc1 and pc2 results
def save_pca_results(pca, embedding, df, group_col, folder, title_prefix):
    emb_df = pd.DataFrame(embedding, columns=['PC1', 'PC2'])
    emb_df[group_col] = df[group_col].values
    emb_df.to_csv(folder / f"{title_prefix}_{group_col}_embedding.csv", index=False)

    # Save variance (How much of the variances will be explained by pc)
    var_df = pd.DataFrame({
        'PC': [f'PC{i + 1}' for i in range(len(pca.explained_variance_ratio_))],
        'ExplainedVarianceRatio': pca.explained_variance_ratio_
    })
    var_df.to_csv(folder / f"{title_prefix}_{group_col}_explained_variance.csv", index=False)

###PCA settings
def run_pca_and_plot(df, group_col, folder, title_prefix, color_map,
                     n_components=2, spot_size=5, save_results=True,
                     svd_solver='auto', whiten=False):


    mz_cols = [c for c in df.columns if "m.z." in c.lower()]
    if not mz_cols:
        raise ValueError("No m/z columns !!!!")

    # Standardize data
    X = StandardScaler().fit_transform(df[mz_cols].fillna(0).values)

    # PCA
    pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten, random_state=42)
    embedding = pca.fit_transform(X)

    if save_results:
        save_pca_results(pca, embedding, df, group_col, folder, title_prefix)

    # Print explained variance for PC1 & PC2
    print(f"{title_prefix} - {group_col} PCA explained variance:")
    for i, var in enumerate(pca.explained_variance_ratio_[:n_components]):
        print(f"  PC{i + 1}: {var:.2%}")
    print(f"  Total for first {n_components} PCs: {pca.explained_variance_ratio_[:n_components].sum():.2%}\n")

    # Plot
    groups = df[group_col].astype(str).values
    if group_col.lower() == "fixation_time":
        unique_groups = sorted(np.unique(groups), key=lambda x: int(x.replace("H", "")))
    else:
        unique_groups = np.unique(groups)

    plt.figure(figsize=(8, 6))
    for grp in unique_groups:
        idx = (groups == grp)
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            label=str(grp),
            s=spot_size * 0.6,
            alpha=0.5,
            color=color_map.get(grp, "#808080")
        )

    plt.legend(markerscale=4, scatterpoints=1)
    plt.title(f"{title_prefix} - colored by {group_col}: slide: 266 & 267 ") #slide: 266 and 267
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(folder / f"{title_prefix}_{group_col}.png", dpi=300)
    plt.close()

    return embedding


# Run PCA
run_pca_and_plot(df_gly, "tissue_type", folders["glycan_tissue"], "Glycan", tissue_colors, spot_size=3)
run_pca_and_plot(df_gly, "fixation_time", folders["glycan_fixation"], "Glycan", fixation_colors, spot_size=3)

run_pca_and_plot(df_pep, "tissue_type", folders["peptide_tissue"], "Peptide", tissue_colors, spot_size=3)
run_pca_and_plot(df_pep, "fixation_time", folders["peptide_fixation"], "Peptide", fixation_colors, spot_size=3)
