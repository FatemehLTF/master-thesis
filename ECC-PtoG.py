#Libraries
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path
import random


###################################################################
# Reference images
GLYCAN_REF_MZ = "m.z.sum_5"
PEPTIDE_REF_MZ = "m.z.sum_5"

# Paths and output folders
BASE_FOLDER = Path("S:/266_and_267/P_G_Alignment")
GLYCAN_CSV = BASE_FOLDER / "Glycan_df.csv"
PEPTIDE_CSV = BASE_FOLDER / "Peptide_df.csv"
OUTPUT_GLYCAN = BASE_FOLDER / "Glycan_Ion_Images_Blue"
OUTPUT_PEPTIDE = BASE_FOLDER / "Peptide_Ion_Images_Red"
OUTPUT_GLYCAN.mkdir(exist_ok=True)
OUTPUT_PEPTIDE.mkdir(exist_ok=True)

#########################################################################
# Load and preprocess dataframes
# loading dfs
df_gly = pd.read_csv(GLYCAN_CSV) #Glycan df
df_pep = pd.read_csv(PEPTIDE_CSV) #Peptide df
# Filter out matrix spots
df_gly = df_gly[~df_gly["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
df_pep = df_pep[~df_pep["tissue_type"].astype(str).str.contains("matrix", case=False, na=False)]
# Filter and keep just only one slide (266 or 267)
df_gly = df_gly[df_gly["Experiment"] == 266]
df_pep = df_pep[df_pep["Experiment"] == 266]
#############################################################################
# Creating ref m/z : sum 5 selected m/z
#Peptide
pep_mz_columns = [col for col in df_pep.columns if col.startswith('m.z.')]
print(f"Found {len(pep_mz_columns)} m/z columns")

random.seed(45)
selected_pep_mz = random.sample(pep_mz_columns, 5)
print("\nPeptide: Selected 5 m/z for reference:")
for i, col in enumerate(selected_pep_mz, 1):
    print(f" {i}. {col}")
df_pep["m.z.sum_5"] = df_pep[selected_pep_mz].sum(axis=1)
#Glycan
gly_mz_columns = [col for col in df_gly.columns if col.startswith('m.z.')]
print(f"Found {len(gly_mz_columns)} m/z columns")

random.seed(42)
selected_gly_mz = random.sample(gly_mz_columns, 5)
print("\nGlycan: Selected 5 m/z for reference:")
for i, col in enumerate(selected_gly_mz, 1):
    print(f" {i}. {col}")
df_gly["m.z.sum_5"] = df_gly[selected_gly_mz].sum(axis=1)

#############################################################################
# Shift coordinates to start from (0,0)
# Glycan
gly_xmin = df_gly["x"].min()
gly_ymin = df_gly["y"].min()
df_gly["x"] = df_gly["x"] - gly_xmin
df_gly["y"] = df_gly["y"] - gly_ymin

# Peptide
pep_xmin = df_pep["x"].min()
pep_ymin = df_pep["y"].min()
df_pep["x"] = df_pep["x"] - pep_xmin
df_pep["y"] = df_pep["y"] - pep_ymin

print(f"Glycan shifted: X:({gly_xmin:.1f}, Y: {gly_ymin:.1f}) --> (0, 0)")
print(f"Peptide shifted: X:({pep_xmin:.1f}, Y: {pep_ymin:.1f}) --> (0, 0)")

# Save shifted tables
df_gly.to_csv(BASE_FOLDER / "glycan_df_shifted.csv", index=False)
df_pep.to_csv(BASE_FOLDER / "peptide_df_shifted.csv", index=False)

##########################################################################
##########################################################################
#### Image settings:
PIXELS_PER_UNIT = 0.05  # desired resolution
SCALE_DOWN = 1     # scale factor
gly_xmin, gly_xmax = df_gly['x'].min(), df_gly['x'].max()
gly_ymin, gly_ymax = df_gly['y'].min(), df_gly['y'].max()
pep_xmin, pep_xmax = df_pep['x'].min(), df_pep['x'].max()
pep_ymin, pep_ymax = df_pep['y'].min(), df_pep['y'].max()

box_xmin = min(gly_xmin, pep_xmin)
box_xmax = max(gly_xmax, pep_xmax)
box_ymin = min(gly_ymin, pep_ymin)
box_ymax = max(gly_ymax, pep_ymax)

# Compute scaled image size
IMG_W = int(np.ceil((box_xmax - box_xmin) * PIXELS_PER_UNIT / SCALE_DOWN))
IMG_H = int(np.ceil((box_ymax - box_ymin) * PIXELS_PER_UNIT / SCALE_DOWN))

print(f"Calculated image size: IMG_W={IMG_W} & IMG_H={IMG_H}")
#############################
#other settings for imaging
SHARP = 1
SIGMA_SCALE = 0.0005
MAX_ITER = 5000
EPS = 1e-7
############################
#####################################################################
# Create reference images for ECC alignment
def make_ref_image(df, mz_col, name_for_save):
    x = df['x'].astype(float).values
    y = df['y'].astype(float).values
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    vals = pd.to_numeric(df.loc[ok, mz_col], errors='coerce').fillna(0).values
    if vals.max() == 0:
        raise ValueError(f"No signal in {mz_col}")

    rw, rh = IMG_W * SHARP, IMG_H * SHARP
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    scale = min(rw / (xmax - xmin + 1e-9), rh / (ymax - ymin + 1e-9))

    px = np.clip(((x - xmin) * scale).astype(int), 0, rw - 1)
    py = np.clip(((y - ymin) * scale).astype(int), 0, rh - 1)

    img = np.zeros((rh, rw), dtype=np.float32)
    np.add.at(img, (py, px), vals)

    img = gaussian_filter(img, sigma=max(img.shape) * SIGMA_SCALE)
    #if img.max() > 0:
        #img /= img.max()

    img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)
    gray8 = (img_resized * 255).astype(np.uint8)

    # Save color reference image
    color = "#0066ff" if "glycan" in name_for_save.lower() else "#ff0000"
    cmap = LinearSegmentedColormap.from_list("", ["white", color])
    rgb = (cmap(img_resized)[:, :, :3] * 255).astype(np.uint8)
    rgb[img_resized == 0] = 255

    Image.fromarray(rgb).save(BASE_FOLDER / f"REF_{name_for_save}_{mz_col.replace('/', '_')}.png")

    return gray8, img_resized.copy(), xmin, xmax, ymin, ymax, scale


fixed_gray8, fixed_float, gxmin, gxmax, gymin, gymax, gscale = make_ref_image(df_gly, GLYCAN_REF_MZ, "Glycan")
moving_gray8, moving_float, pxmin, pxmax, pymin, pymax, pscale = make_ref_image(df_pep, PEPTIDE_REF_MZ, "Peptide")

############################################################################
# ECC alignment with inverse matrix calculation
fixed_ecc = cv2.GaussianBlur(fixed_float.astype(np.float32), (5, 5), 0.8)
moving_ecc = cv2.GaussianBlur(moving_float.astype(np.float32), (5, 5), 0.8)

# Initialize affine for warp vector
warp_forward = np.eye(2, 3, dtype=np.float32)  # peptide → glycan
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, MAX_ITER, EPS)

# Run ECC
cc, warp_forward = cv2.findTransformECC(
    fixed_ecc, moving_ecc, warp_forward,
    motionType=cv2.MOTION_AFFINE,
    criteria=criteria,
    inputMask=None,
    gaussFiltSize=1
)

# Compute INVERSE warp vector
a, b, tx = warp_forward[0]
c, d, ty = warp_forward[1]
det = a * d - b * c

warp_inverse = np.array([
    [d / det, -b / det, (b * ty - d * tx) / det],
    [-c / det, a / det, (c * tx - a * ty) / det]
], dtype=np.float32)

# Save transformation matrices
np.savetxt(BASE_FOLDER / "warp_matrix_peptide_to_glycan_FORWARD.csv",
           warp_forward, delimiter=",", fmt="%.10f")
np.savetxt(BASE_FOLDER / "warp_matrix_glycan_to_peptide_INVERSE.csv",
           warp_inverse, delimiter=",", fmt="%.10f")

#######################################
#REF overlay img
# Create alignment verification overlay using forward warp (using refrence images)
aligned_check = cv2.warpAffine(
    moving_gray8,
    warp_forward,
    (IMG_W, IMG_H),
    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
)
overlay = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
overlay[:, :, 2] = fixed_gray8  # Blue = glycan
overlay[:, :, 0] = aligned_check  # Red = warped peptide
Image.fromarray(overlay).save(BASE_FOLDER / "REF_OVERLAY_CHECK.png")

###########################################################################
# Create scaling matrices to convert between coordinate and pixel spaces
S_pep = np.array([
    [pscale, 0, -pxmin * pscale],
    [0, pscale, -pymin * pscale],
    [0, 0, 1]
])

S_gly_inv = np.array([
    [1/gscale, 0, gxmin],
    [0, 1/gscale, gymin],
    [0, 0, 1]
])

# Combine transformations
# Peptide coordinates → Peptide pixels → ECC transformation → Glycan pixels → Glycan coordinates
ECC_homogeneous = np.vstack([warp_inverse, [0, 0, 1]])
T_total = S_gly_inv @ ECC_homogeneous @ S_pep
T_affine = T_total[:2, :]

# Apply corrected transformation
df_pep['x_aligned'] = T_affine[0, 0] * df_pep['x'] + T_affine[0, 1] * df_pep['y'] + T_affine[0, 2]
df_pep['y_aligned'] = T_affine[1, 0] * df_pep['x'] + T_affine[1, 1] * df_pep['y'] + T_affine[1, 2]


# Compute aligned peptide bounds
valid = np.isfinite(df_pep['x_aligned']) & np.isfinite(df_pep['y_aligned'])
aligned_xmin = df_pep['x_aligned'][valid].min()
aligned_xmax = df_pep['x_aligned'][valid].max()
aligned_ymin = df_pep['y_aligned'][valid].min()
aligned_ymax = df_pep['y_aligned'][valid].max()

##############################################################################
# Compute bounding box for consistent image generation
box_xmin = min(gxmin, aligned_xmin)
box_xmax = max(gxmax, aligned_xmax)
box_ymin = min(gymin, aligned_ymin)
box_ymax = max(gymax, aligned_ymax)

# Calculate scale based on box bounds
rw, rh = IMG_W * SHARP, IMG_H * SHARP
box_scale = min(rw / (box_xmax - box_xmin + 1e-9),
                  rh / (box_ymax - box_ymin + 1e-9))

box_ref_params = (box_xmin, box_xmax, box_ymin, box_ymax, box_scale)

print(f"box bounds: x {box_xmin:.1f}..{box_xmax:.1f}, y {box_ymin:.1f}..{box_ymax:.1f}")
print(f"box scale: {box_scale:.6f}")
print(f"Glycan scale was: {gscale:.6f}")
print(f"Peptide original scale was: {pscale:.6f}")

##############################################################################
# Generate all ion images using the bounding box
def generate_all_ions(df, folder, color_hex, use_aligned=False, ref_params=None):
    mz_cols = [c for c in df.columns if any(p in c.lower() for p in ["m/z", "m.z.", "mz"])]
    if not mz_cols:
        mz_cols = df.columns[10:]

    x_col = 'x_aligned' if use_aligned else 'x'
    y_col = 'y_aligned' if use_aligned else 'y'
    cmap = LinearSegmentedColormap.from_list("", ["white", color_hex])

    xmin, xmax, ymin, ymax, scale = ref_params
    rw, rh = IMG_W * SHARP, IMG_H * SHARP

    for idx, col in enumerate(mz_cols, 1):
        x = df[x_col].astype(float).values
        y = df[y_col].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        vals = pd.to_numeric(df.loc[ok, col], errors='coerce').fillna(0).values

        if vals.max() == 0:
            continue

        # Convert coordinates to pixel positions using box bounds
        px = np.clip(((x[ok] - xmin) * scale).astype(int), 0, rw - 1)
        py = np.clip(((y[ok] - ymin) * scale).astype(int), 0, rh - 1)

        img = np.zeros((rh, rw), dtype=np.float32)
        np.add.at(img, (py, px), vals)

        img = gaussian_filter(img, sigma=max(img.shape) * SIGMA_SCALE)
        if img.max() > 0:
            img /= img.max()

        img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)
        rgb = (cmap(img_resized)[:, :, :3] * 255).astype(np.uint8)
        rgb[img_resized == 0] = 255

        safe_name = "".join(c if c.isalnum() or c in "._-+" else "_" for c in str(col))
        Image.fromarray(rgb).save(folder / f"{safe_name}.png")

        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(mz_cols)} images for {folder.name}")


generate_all_ions(df_gly, OUTPUT_GLYCAN, "#0066ff", ref_params=box_ref_params)
generate_all_ions(df_pep, OUTPUT_PEPTIDE, "#ff0000", use_aligned=True, ref_params=box_ref_params)

##############################################################################
# Save aligned peptide dataframe
df_pep.to_csv(BASE_FOLDER / "peptide_df_aligned_to_glycan.csv", index=False)
