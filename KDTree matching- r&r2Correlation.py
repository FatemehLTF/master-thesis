# Matching peptide and glycan by KDTree
# peptide_glycan_correlation r and r2

#Run seprataly for each slide : once for slide 266 and once for 267

### libraries
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

### Input paths
BASE_FOLDER = Path(r"S:/266_and_267/final_P_G_Alignment\HM_266")
PEPTIDE_CSV = BASE_FOLDER / "peptide_df_aligned_to_glycan.csv"
GLYCAN_CSV = BASE_FOLDER / "glycan_df_shifted.csv"
### Output paths
OUTPUT_CORR = BASE_FOLDER / "corr.csv"
OUTPUT_PVAL = BASE_FOLDER / "pvalues.csv"
OUTPUT_HEATMAP = BASE_FOLDER / "heatmap.png"
OUTPUT_HEATMAP_SIG = BASE_FOLDER / "heatmap_significance.png"
OUTPUT_HEATMAP_EFFECT = BASE_FOLDER / "heatmap_significance_Effect_size.png"

###Functions:
# m/z column naming : m.z.XXX:
def extract_mz(col_name):
    import re
    numbers = re.findall(r'\d+\.\d+|\d+', str(col_name))
    return float(numbers[0]) if numbers else 0.0

######################################################################
#Loading DFs
df_pep = pd.read_csv(PEPTIDE_CSV)
df_gly = pd.read_csv(GLYCAN_CSV)
######################################################################
# Filter out matrix spots
df_gly = df_gly[~df_gly["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
df_pep = df_pep[~df_pep["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
# Filter and keep just only one slide (267 or 266)
df_gly = df_gly[df_gly["Experiment"] == 266]
df_pep = df_pep[df_pep["Experiment"] == 266]
# Coordinates:
pep_x, pep_y = ('x_aligned' if 'x_aligned' in df_pep else 'x',
                'y_aligned' if 'y_aligned' in df_pep else 'y')

# Identify m/z columns: "m.z.XXX"
pep_cols = [c for c in df_pep.columns if c.lower().startswith('m.z.')]
gly_cols = [c for c in df_gly.columns if c.lower().startswith('m.z.')]

if len(pep_cols) == 0 or len(gly_cols) == 0:
    raise ValueError("No m/z columns found!")

# Convert all m/z columns to numeric
for col in pep_cols:
    df_pep[col] = pd.to_numeric(df_pep[col], errors='coerce')
for col in gly_cols:
    df_gly[col] = pd.to_numeric(df_gly[col], errors='coerce')

# Keep only coordinates  and m/z columns
df_pep = df_pep[[pep_x, pep_y]+pep_cols]
df_gly = df_gly[['x','y']+gly_cols]

#######################################################################
#######################################################################
# KDTree: Spot matching by adaptive Threshold
# KDTree is used to efficiently find the nearest glycan spot for each peptide spot
pep_coords = df_pep[[pep_x, pep_y]].values
gly_coords = df_gly[['x','y']].values
tree = KDTree(gly_coords)
distances, indices = tree.query(pep_coords, k=1)
#distances: Distance to nearest glycan
#indices: Index of nearest glycan

###############################
# check distances
min_distance = np.min(distances)
max_distance = np.max(distances)
median_distance = np.median(distances)

print(f"Minimum distance: {min_distance:.3f} µm")
print(f"Maximum distance: {max_distance:.3f} µm")
print(f"Median distance: {median_distance:.3f} µm")

##############################
# Adaptive threshold: only matches spots that are real close!

median_dist = np.median(distances)
#threshold = median_dist * 2
#threshold = 28.3/2  # in µm
threshold = (28.3*2) / 3  # µm
mask = distances <= threshold
df_pep_matched = df_pep[mask].reset_index(drop=True)
df_gly_matched = df_gly.iloc[indices[mask]].reset_index(drop=True)

print(f"Threshold used: {threshold:.3f}")
print(f"median_dist: {median_dist:.3f}")
print(f"Matched spots: {len(df_pep_matched)} / {len(df_pep)}")


########################################################################
# CORRELATION ANALYSIS

# empty matrices for correlations
corr_matrix = pd.DataFrame(index=pep_cols, columns=gly_cols, dtype=float)
# empty matrices for p-values
p_matrix = pd.DataFrame(index=pep_cols, columns=gly_cols, dtype=float)

gly_data_dict = {col: df_gly_matched[col].values for col in gly_cols}
# Computes Pearson correlation
for pep_col in pep_cols:
    pep_data = df_pep_matched[pep_col].values
    mask_valid = ~np.isnan(pep_data)
    for gly_col in gly_cols:
        gly_data = gly_data_dict[gly_col]
        mask = mask_valid & ~np.isnan(gly_data)
        if mask.sum() > 10:  # minimum valid samples
            try:
                r, p = pearsonr(pep_data[mask], gly_data[mask])
                corr_matrix.loc[pep_col, gly_col] = r
                p_matrix.loc[pep_col, gly_col] = p
            except:
                corr_matrix.loc[pep_col, gly_col] = np.nan
                p_matrix.loc[pep_col, gly_col] = np.nan

# Save matrices
corr_matrix.to_csv(OUTPUT_CORR)
p_matrix.to_csv(OUTPUT_PVAL)

######################################################################
######################################################################
### Heatmap Visualization
n_rows, n_cols = corr_matrix.shape
fig_width = max(12, n_cols * 0.3)
fig_height = max(10, n_rows * 0.3)

##########################
# Plot 1: Heatmap
plt.figure(figsize=(fig_width, fig_height))
ax1 = sns.heatmap(
    corr_matrix.astype(float),
    cmap='RdBu_r',
    center=0,
    xticklabels=[f"{extract_mz(c):.1f}" for c in corr_matrix.columns],
    yticklabels=[f"{extract_mz(c):.1f}" for c in corr_matrix.index],
    cbar_kws={"label": "Pearson r"}
)
ax1.set_xlabel("Glycan m/z", fontsize=40)
ax1.set_ylabel("Peptide m/z", fontsize=40)
ax1.set_title(f"Peptide–Glycan Correlation ({len(df_pep_matched)} matched spots)", fontsize=30)
ax1.tick_params(axis='x', rotation=90, labelsize=20)
ax1.tick_params(axis='y', rotation=0, labelsize=20)

colorbar1 = ax1.collections[0].colorbar
colorbar1.ax.tick_params(labelsize=40)
colorbar1.set_label("Pearson r", fontsize=40)
# Overlay stars
for i in range(n_rows):
    for j in range(n_cols):
        r_value = corr_matrix.iloc[i, j]
        if r_value > 0.3:
            ax1.text(j + 0.5, i + 0.5, "★", ha='center', va='center', color='black', fontsize=20, fontweight='bold')
        elif r_value < -0.3:
            ax1.text(j + 0.5, i + 0.5, "☆", ha='center', va='center', color='black', fontsize=20, fontweight='bold')


plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP, dpi=300)
plt.close()

##########################
# Plot 2: Heatmap of Effect Size: r2
effect_matrix = corr_matrix**2
plt.figure(figsize=(fig_width, fig_height))
ax3 = sns.heatmap(
    effect_matrix.astype(float),
    cmap='coolwarm',
    center=0,
    xticklabels=[f"{extract_mz(c):.1f}" for c in effect_matrix.columns],
    yticklabels=[f"{extract_mz(c):.1f}" for c in effect_matrix.index],
    cbar_kws={"label": "Effect Size"}
)
ax3.set_xlabel("Glycan m/z", fontsize=40)
ax3.set_ylabel("Peptide m/z", fontsize=40)
ax3.set_title(f"Peptide–Glycan Effect Size ({len(df_pep_matched)} matched spots)", fontsize=30)
ax3.tick_params(axis='x', rotation=90, labelsize=20)
ax3.tick_params(axis='y', rotation=0, labelsize=20)

colorbar3 = ax3.collections[0].colorbar
colorbar3.ax.tick_params(labelsize=40)
colorbar3.set_label("Effect Size", fontsize=40)

for i in range(effect_matrix.shape[0]):
    for j in range(effect_matrix.shape[1]):
        if effect_matrix.iloc[i, j] > 0.15:
            ax3.text(j + 0.5, i + 0.5, "★", ha='center', va='center', color='black', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_HEATMAP_EFFECT, dpi=300)
plt.close()

######################################################################
###CHECK: Spot Matching
# All spots (before matching)
fig1, ax1 = plt.subplots(figsize=(24,8))
ax1.scatter(df_pep[pep_x], df_pep[pep_y], s=1, alpha=0.5, c='red', label='Peptide')
ax1.scatter(df_gly['x'], df_gly['y'], s=1, alpha=0.5, c='blue', label='Glycan')
ax1.set_title(f'All Spots\nPeptide: {len(df_pep)} | Glycan: {len(df_gly)}', fontsize=20)
ax1.set_xlabel('X coordinate', fontsize=16)
ax1.set_ylabel('Y coordinate', fontsize=16)
ax1.legend(fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_FOLDER / "all_spots_comparison_small.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# Matched spots only
fig2, ax2 = plt.subplots(figsize=(24, 8))
ax2.scatter(df_pep_matched[pep_x], df_pep_matched[pep_y], s=1, alpha=0.5, c='red', label='Peptide')
ax2.scatter(df_gly_matched['x'], df_gly_matched['y'], s=1, alpha=0.5, c='blue', label='Glycan')
ax2.set_title(f'Matched Spots Only\nCount: {len(df_pep_matched)}', fontsize=20)
ax2.set_xlabel('X coordinate', fontsize=16)
ax2.set_ylabel('Y coordinate', fontsize=16)
ax2.legend(fontsize=14)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_FOLDER / "matched_spots_only_small.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

# All spots (before matching)
fig1, ax1 = plt.subplots(figsize=(120,40))
ax1.scatter(df_pep[pep_x], df_pep[pep_y], s=1, alpha=0.5, c='red', label='Peptide')
ax1.scatter(df_gly['x'], df_gly['y'], s=1, alpha=0.5, c='blue', label='Glycan')
ax1.set_title(f'All Spots\nPeptide: {len(df_pep)} | Glycan: {len(df_gly)}', fontsize=20)
ax1.set_xlabel('X coordinate', fontsize=16)
ax1.set_ylabel('Y coordinate', fontsize=16)
ax1.legend(fontsize=14)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_FOLDER / "all_spots_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# Matched spots only
fig2, ax2 = plt.subplots(figsize=(120, 40))
ax2.scatter(df_pep_matched[pep_x], df_pep_matched[pep_y], s=1, alpha=0.5, c='red', label='Peptide')
ax2.scatter(df_gly_matched['x'], df_gly_matched['y'], s=1, alpha=0.5, c='blue', label='Glycan')
ax2.set_title(f'Matched Spots Only\nCount: {len(df_pep_matched)}', fontsize=20)
ax2.set_xlabel('X coordinate', fontsize=16)
ax2.set_ylabel('Y coordinate', fontsize=16)
ax2.legend(fontsize=14)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_FOLDER / "matched_spots_only.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

# Show connection lines for all spots
n_to_show = min(93561, len(df_pep_matched))
fig3, ax3 = plt.subplots(figsize=(120, 40))
ax3.scatter(df_pep_matched[pep_x].iloc[:n_to_show],
            df_pep_matched[pep_y].iloc[:n_to_show],
            s=10, alpha=0.7, c='red', label='Peptide')
ax3.scatter(df_gly_matched['x'].iloc[:n_to_show],
            df_gly_matched['y'].iloc[:n_to_show],
            s=10, alpha=0.7, c='blue', label='Glycan')
# connection lines
for i in range(n_to_show):
    ax3.plot([df_pep_matched[pep_x].iloc[i], df_gly_matched['x'].iloc[i]],
             [df_pep_matched[pep_y].iloc[i], df_gly_matched['y'].iloc[i]],
             'k-', alpha=0.2, linewidth=0.5)

ax3.set_title(f'Match Connections (First {n_to_show} spots)\nMedian Distance: {median_dist:.2f}', fontsize=20)
ax3.set_xlabel('X coordinate', fontsize=16)
ax3.set_ylabel('Y coordinate', fontsize=16)
ax3.legend(fontsize=14)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BASE_FOLDER / "match_connections.png", dpi=300, bbox_inches='tight')
plt.close(fig3)
#################################################################################################
# Before Matching
n_before = min(93561, len(df_pep))
df_gly_before = df_gly.iloc[indices].reset_index(drop=True)

fig_before, ax_before = plt.subplots(figsize=(120, 40))
# Plot all peptides (before filtering)
ax_before.scatter(
    df_pep[pep_x].iloc[:n_before],
    df_pep[pep_y].iloc[:n_before],
    s=10, alpha=0.7, c='red', label='Peptide (all)'
)

# Plot all glycan nearest matches (before filtering)
ax_before.scatter(
    df_gly_before['x'].iloc[:n_before],
    df_gly_before['y'].iloc[:n_before],
    s=10, alpha=0.7, c='blue', label='Nearest Glycan (all)'
)

# Draw ALL connection lines (before threshold)
for i in range(n_before):
    ax_before.plot(
        [df_pep[pep_x].iloc[i], df_gly_before['x'].iloc[i]],
        [df_pep[pep_y].iloc[i], df_gly_before['y'].iloc[i]],
        'k-', alpha=0.15, linewidth=0.5
    )

ax_before.set_title(
    f'Connections Before Matching (First {n_before} spots)\n'
    f'Median Distance: {median_dist:.2f} µm',
    fontsize=20
)
ax_before.set_xlabel('X coordinate', fontsize=16)
ax_before.set_ylabel('Y coordinate', fontsize=16)
ax_before.legend(fontsize=14)
ax_before.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE_FOLDER / "match_connections_before_threshold.png",
            dpi=300, bbox_inches='tight')
plt.close(fig_before)

################################################################################################
# Histogram of all distances
bin_width = 5  # µm
num_bins = int(np.ceil(distances.max() / bin_width))
counts, bins, patches = plt.hist(distances, bins=num_bins, color='skyblue', edgecolor='black')

plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold} µm')
plt.title("Peptide-Glycan Distances (Before Threshold, zoom 0-100 µm)")
plt.xlabel("Distance (µm)")
plt.ylabel("Number of Peptides")

plt.xlim(0, 100)

for count, patch in zip(counts, patches):
    x = patch.get_x()
    if x >= 0 and x <= 100 and count > 0:
        plt.text(x + patch.get_width()/2, count, int(count),
                 ha='center', va='bottom', fontsize=8)

plt.legend()
plt.tight_layout()
plt.savefig(BASE_FOLDER / "distance_before_threshold_zoom.png", dpi=300)
plt.close()


##############################################################################

plt.figure(figsize=(120, 40))
plt.scatter(df_pep[pep_x], df_pep[pep_y], s=10, c='red', alpha=0.5, label='Peptide')
plt.scatter(df_gly['x'], df_gly['y'], s=10, c='blue', alpha=0.5, label='Glycan')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.title("All Peptide (red) and Glycan (blue) spots - BEFORE matching")
plt.legend()
plt.tight_layout()
plt.savefig(BASE_FOLDER / "all_peptide_glycan_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
