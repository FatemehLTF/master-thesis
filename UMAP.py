# Umap plots

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt

# Paths to input fildes and output folders
BASE_FOLDER = Path("S:/266_and_267/UMAP")
GLYCAN_CSV = BASE_FOLDER / "Glycan_df.csv"
PEPTIDE_CSV = BASE_FOLDER / "Peptide_df.csv"

UMAP_OUTPUT = BASE_FOLDER / "UMAP_Plots_both"
UMAP_OUTPUT.mkdir(exist_ok=True, parents=True)

folders = {
    "glycan_tissue": UMAP_OUTPUT / "Glycan_Tissue",
    "glycan_fixation": UMAP_OUTPUT / "Glycan_Fixation",
    "peptide_tissue": UMAP_OUTPUT / "Peptide_Tissue",
    "peptide_fixation": UMAP_OUTPUT / "Peptide_Fixation"
}
for f in folders.values():
    f.mkdir(exist_ok=True)

# Load datasets
df_gly = pd.read_csv(GLYCAN_CSV)
df_pep = pd.read_csv(PEPTIDE_CSV)

# Remove matrix spots and filter slide 266 or 267 ( -#- used to keep both)
# Repeated for each slide also both-slides-mixed (3 times Run) : 2 times run separate for each slide number
df_gly = df_gly[~df_gly["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
df_pep = df_pep[~df_pep["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
#df_gly = df_gly[df_gly["Experiment"] == 267]
#df_pep = df_pep[df_pep["Experiment"] == 267]

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
# for fixation groups
fixation_colors = {
    "3H": "#FF00FF",
    "6H": "#FFFF00",
    "12H": "#0000FF",
    "24H": "#FFA500",
    "60H": "#00FF00"
}

def save_umap_results(reducer, embedding, df, group_col, folder, title_prefix):
    emb_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    emb_df[group_col] = df[group_col].values
    emb_df.to_csv(folder / f"{title_prefix}_{group_col}_embedding.csv", index=False)
 #UMAP setting; Run base euclidean
def run_umap_and_plot(df, group_col, folder, title_prefix, color_map,
                      n_neighbors=15, min_dist=0.1, metric='euclidean',
                      spot_size=5, save_results=True):
    
#m/z columns naming like : m.z.856.365
    mz_cols = [c for c in df.columns if "m.z." in c.lower()]
    if not mz_cols:
        raise ValueError("No m/z columns")

    X = StandardScaler().fit_transform(df[mz_cols].fillna(0).values)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=42)
    embedding = reducer.fit_transform(X)

    if save_results:
        save_umap_results(reducer, embedding, df, group_col, folder, title_prefix)

    groups = df[group_col].astype(str).values

    if group_col.lower() == "fixation_time":
        unique_groups = sorted(np.unique(groups), key=lambda x: int(x.replace("H", "")))
    else:
        unique_groups = np.unique(groups)

#plot image settings
    plt.figure(figsize=(8, 6))
    for grp in unique_groups:
        idx = (groups == grp)
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            label=str(grp),
            s=spot_size * 0.6,  # size of spots
            alpha=0.5,  # transparent
            color=color_map.get(grp, "#808080")
        )

    plt.legend(markerscale=4, scatterpoints=1)
    plt.title(f"{title_prefix} - colored by {group_col}: slide: 266 & 267 (both)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(folder / f"{title_prefix}_{group_col}.png", dpi=300)
    plt.close()

    return embedding

# Run UMAP function for each condition
run_umap_and_plot(df_gly, "tissue_type", folders["glycan_tissue"], "Glycan", tissue_colors, spot_size=3)
run_umap_and_plot(df_gly, "fixation_time", folders["glycan_fixation"], "Glycan", fixation_colors, spot_size=3)

run_umap_and_plot(df_pep, "tissue_type", folders["peptide_tissue"], "Peptide", tissue_colors, spot_size=3)
run_umap_and_plot(df_pep, "fixation_time", folders["peptide_fixation"], "Peptide", fixation_colors, spot_size=3)
