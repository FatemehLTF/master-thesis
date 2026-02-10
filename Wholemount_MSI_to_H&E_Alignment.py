# MSI to H&E Alignment using ECC
import cv2
import numpy as np
import pandas as pd
import random
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path

# Paths to datasets
BASE_FOLDER = Path(r"S:\H&E_wholemount")
HE_CSV = BASE_FOLDER / "H&E_pixels.csv"
# H&E_pixels.csv data set creating using "H&E convert to vector" script
PEPTIDE_CSV = BASE_FOLDER / "all_data_df_PEP.csv"
OUTPUT_PEPTIDE_YELLOW_TRANSPARENT = BASE_FOLDER

# Output folders
OUTPUT_HE = BASE_FOLDER / "HE_Optical_Images_Blue"
OUTPUT_PEPTIDE = BASE_FOLDER / "Peptide_Ion_Images_Red"
OUTPUT_HE.mkdir(exist_ok=True)
OUTPUT_PEPTIDE.mkdir(exist_ok=True)

### Loading data frames
df_he = pd.read_csv(HE_CSV)
df_pep = pd.read_csv(PEPTIDE_CSV)
#columns name in peptide data set : e.g. 'm.z. 500.500'
mz_columns = [col for col in df_pep.columns if col.startswith('m.z.')]
print(f"Found {len(mz_columns)} m/z columns")
if len(mz_columns) < 5:
    raise ValueError(f"Need at least 5 m/z columns")

random.seed(45) #get similar result after each run!
selected_mz = random.sample(mz_columns, 10)
print("\nSelected 20 m/z for reference:")
for i, col in enumerate(selected_mz, 1):
    print(f" {i}. {col}")

df_pep["ref_mz_sum"] = df_pep[selected_mz].sum(axis=1)

print(f"H&E loaded: {len(df_he):,} pixels (entire image)")
print(f"Peptide spots: {len(df_pep)}")

# Shift coordinates (bring both in similar space)
he_xmin, he_ymin = df_he["x"].min(), df_he["y"].min()
df_he["x_shifted"] = df_he["x"] - he_xmin
df_he["y_shifted"] = df_he["y"] - he_ymin

pep_xmin, pep_ymin = df_pep["x"].min(), df_pep["y"].min()
df_pep["x_shifted"] = df_pep["x"] - pep_xmin
df_pep["y_shifted"] = df_pep["y"] - pep_ymin


# Image settings

PIXELS_PER_UNIT = 0.2  # ~5 pixels per µm
SCALE_DOWN = 1
SHARP = 2  # super-sampling factor
SIGMA_SCALE = 0.001

all_xmin = min(df_he['x_shifted'].min(), df_pep['x_shifted'].min())
all_xmax = max(df_he['x_shifted'].max(), df_pep['x_shifted'].max())
all_ymin = min(df_he['y_shifted'].min(), df_pep['y_shifted'].min())
all_ymax = max(df_he['y_shifted'].max(), df_pep['y_shifted'].max())

IMG_W = int(np.ceil((all_xmax - all_xmin) * PIXELS_PER_UNIT / SCALE_DOWN))
IMG_H = int(np.ceil((all_ymax - all_ymin) * PIXELS_PER_UNIT / SCALE_DOWN))

print(f"Image size: {IMG_W} x {IMG_H} (super-res: {IMG_W*SHARP} x {IMG_H*SHARP})")

# Reference images: H&E inverted (dark tissue = strong blue), clean white background
def make_ref_image(df, intensity_col, name_for_save):
    x = df['x_shifted'].values.astype(float)
    y = df['y_shifted'].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    vals = pd.to_numeric(df.loc[ok, intensity_col], errors='coerce').fillna(0).values
    if vals.max() == 0:
        raise ValueError(f"No signal in {intensity_col}")

    rw, rh = IMG_W * SHARP, IMG_H * SHARP
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    scale = min(rw / (xmax - xmin + 1e-9), rh / (ymax - ymin + 1e-9))

    px = np.clip(((x - xmin) * scale).astype(int), 0, rw - 1)
    py = np.clip(((y - ymin) * scale).astype(int), 0, rh - 1)

    img = np.zeros((rh, rw), dtype=np.float32)
    np.add.at(img, (py, px), vals)

    sigma = max(rw, rh) * SIGMA_SCALE
    img = gaussian_filter(img, sigma=sigma)
    img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)
    gray8 = np.clip(img_resized * 255, 0, 255).astype(np.uint8)

    # Colored reference
    color = "#000000" if "he" in name_for_save.lower() else "#006400"
    cmap = LinearSegmentedColormap.from_list("", ["white", color])
    display_img = img_resized.copy()

    # INVERT ONLY FOR H&E: dark staining (low intensity)
    if "he" in name_for_save.lower():
        if display_img.max() > 0:
            display_img = display_img.max() - display_img  # Invert

    # Clip outliers and normalize
    if display_img.max() > 0:
        p99 = np.percentile(display_img[display_img > 0], 99.9)
        display_img = np.clip(display_img, 0, p99)
        display_img /= display_img.max()

    rgb = (cmap(display_img)[:, :, :3] * 255).astype(np.uint8)

    # white background
    rgb[img_resized == 0] = 255
    low_thresh = img_resized.max() * 0.01
    rgb[img_resized < low_thresh] = 255

    Image.fromarray(rgb).save(BASE_FOLDER / f"REF_{name_for_save}.png")
    return gray8, img_resized.copy(), xmin, xmax, ymin, ymax, scale

### Generating reference images
fixed_gray8, fixed_float, hxmin, hxmax, hymin, hymax, hscale = make_ref_image(df_he, "Intensity", "HE_All_Cores")
moving_gray8, moving_float, pxmin, pxmax, pymin, pymax, pscale = make_ref_image(df_pep, "ref_mz_sum", "Peptide")

print(f"H&E bounds: x {hxmin:.1f}..{hxmax:.1f} y {hymin:.1f}..{hymax:.1f} scale {hscale:.6f}")
print(f"Peptide bounds: x {pxmin:.1f}..{pxmax:.1f} y {pymin:.1f}..{pymax:.1f} scale {pscale:.6f}")

############################################################################
# ECC alignment

fixed_ecc = cv2.GaussianBlur(fixed_float.astype(np.float32), (5, 5), 0.8)
moving_ecc = cv2.GaussianBlur(moving_float.astype(np.float32), (5, 5), 0.8)

warp_forward = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)
cc, warp_forward = cv2.findTransformECC(
    fixed_ecc, moving_ecc, warp_forward,
    motionType=cv2.MOTION_AFFINE, criteria=criteria,
    inputMask=None, gaussFiltSize=1)
print(f"ECC correlation: {cc:.4f}")

# Inverse warp calculation
a, b, tx = warp_forward[0]
c, d, ty = warp_forward[1]
det = a * d - b * c
warp_inverse = np.array([
    [d / det, -b / det, (b * ty - d * tx) / det],
    [-c / det, a / det, (c * tx - a * ty) / det]
], dtype=np.float32)

np.savetxt(BASE_FOLDER / "warp_peptide_to_HE_FORWARD.csv", warp_forward, delimiter=",", fmt="%.10f")
np.savetxt(BASE_FOLDER / "warp_peptide_to_HE_INVERSE.csv", warp_inverse, delimiter=",", fmt="%.10f")

# Apply transformation and final bounding box
S_pep = np.array([[pscale, 0, -pxmin * pscale],
                  [0, pscale, -pymin * pscale],
                  [0, 0, 1]])
S_he_inv = np.array([[1/hscale, 0, hxmin],
                     [0, 1/hscale, hymin],
                     [0, 0, 1]])
ECC_hom = np.vstack([warp_inverse, [0, 0, 1]])
T_total = S_he_inv @ ECC_hom @ S_pep
T_affine = T_total[:2, :]

df_pep['x_aligned'] = T_affine[0, 0] * df_pep['x_shifted'] + T_affine[0, 1] * df_pep['y_shifted'] + T_affine[0, 2]
df_pep['y_aligned'] = T_affine[1, 0] * df_pep['x_shifted'] + T_affine[1, 1] * df_pep['y_shifted'] + T_affine[1, 2]

valid = np.isfinite(df_pep['x_aligned']) & np.isfinite(df_pep['y_aligned'])
aligned_xmin = df_pep['x_aligned'][valid].min()
aligned_xmax = df_pep['x_aligned'][valid].max()
aligned_ymin = df_pep['y_aligned'][valid].min()
aligned_ymax = df_pep['y_aligned'][valid].max()

box_xmin = min(hxmin, aligned_xmin)
box_xmax = max(hxmax, aligned_xmax)
box_ymin = min(hymin, aligned_ymin)
box_ymax = max(hymax, aligned_ymax)

box_scale = min((IMG_W * SHARP) / (box_xmax - box_xmin + 1e-9),
                (IMG_H * SHARP) / (box_ymax - box_ymin + 1e-9))
box_ref_params = (box_xmin, box_xmax, box_ymin, box_ymax, box_scale)

print(f"Final aligned bounding box: x {box_xmin:.1f}..{box_xmax:.1f} y {box_ymin:.1f}..{box_ymax:.1f} scale {box_scale:.6f}")

##########################################################
# Generating final images

def generate_all_ions(df, folder, color_hex, intensity_cols=None, use_aligned=False, ref_params=None):
    if intensity_cols is None:
        intensity_cols = [c for c in df.columns if c.startswith('m.z.')]
    if "Intensity" in df.columns and "HE" in folder.name:
        intensity_cols = ["Intensity"]

    x_col = 'x_aligned' if use_aligned else 'x_shifted'
    y_col = 'y_aligned' if use_aligned else 'y_shifted'

    cmap = LinearSegmentedColormap.from_list("", ["white", color_hex])
    xmin, xmax, ymin, ymax, scale = ref_params
    rw, rh = IMG_W * SHARP, IMG_H * SHARP

    for idx, col in enumerate(intensity_cols, 1):
        x = df[x_col].astype(float).values
        y = df[y_col].astype(float).values
        ok = np.isfinite(x) & np.isfinite(y)
        vals = pd.to_numeric(df.loc[ok, col], errors='coerce').fillna(0).values
        if vals.max() == 0:
            continue

        px = np.clip(((x[ok] - xmin) * scale).astype(int), 0, rw - 1)
        py = np.clip(((y[ok] - ymin) * scale).astype(int), 0, rh - 1)

        img = np.zeros((rh, rw), dtype=np.float32)
        np.add.at(img, (py, px), vals)

        sigma = max(rw, rh) * SIGMA_SCALE
        img = gaussian_filter(img, sigma=sigma)
        img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)

        display_img = img_resized.copy()

        # INVERT ONLY FOR H&E
        if "HE" in folder.name:
            if display_img.max() > 0:
                display_img = display_img.max() - display_img
            if display_img.max() > 0:
                display_img /= display_img.max()
        else:
            # Peptide: clip 99.9th percentile
            if display_img.max() > 0:
                p99 = np.percentile(display_img[display_img > 0], 99.9)
                display_img = np.clip(display_img, 0, p99)
                display_img /= display_img.max()

        rgb = (cmap(display_img)[:, :, :3] * 255).astype(np.uint8)
        rgb[img_resized == 0] = 255
        low_thresh = img_resized.max() * 0.01
        rgb[img_resized < low_thresh] = 255

        safe_name = "".join(c if c.isalnum() or c in "._-+" else "_" for c in str(col))
        Image.fromarray(rgb).save(folder / f"{safe_name}.png")

        if idx % 20 == 0 or idx == len(intensity_cols):
            print(f" Saved {idx}/{len(intensity_cols)} → {folder.name}")

generate_all_ions(df_he, OUTPUT_HE, "#000000", intensity_cols=["Intensity"], ref_params=box_ref_params)
generate_all_ions(df_pep, OUTPUT_PEPTIDE, "#006400", use_aligned=True, ref_params=box_ref_params)

################################################################
# Generate true full-color RGB H&E image using the real R, G, B columns
# Coordinates (use shifted, same as H&E reference)
x = df_he['x_shifted'].values.astype(float)
y = df_he['y_shifted'].values.astype(float)
ok = np.isfinite(x) & np.isfinite(y)
x, y = x[ok], y[ok]

# Extract RGB values
r_vals = df_he.loc[ok, 'R'].values.astype(float)
g_vals = df_he.loc[ok, 'G'].values.astype(float)
b_vals = df_he.loc[ok, 'B'].values.astype(float)

r_vals = np.clip(r_vals, 0, 255)
g_vals = np.clip(g_vals, 0, 255)
b_vals = np.clip(b_vals, 0, 255)

# Use the same final bounding box
xmin, xmax, ymin, ymax, scale = box_ref_params
rw, rh = IMG_W * SHARP, IMG_H * SHARP

# Pixel coordinates
px = np.clip(((x - xmin) * scale).astype(int), 0, rw - 1)
py = np.clip(((y - ymin) * scale).astype(int), 0, rh - 1)

# Accumulate each channel separately
img_r = np.zeros((rh, rw), dtype=np.float32)
img_g = np.zeros((rh, rw), dtype=np.float32)
img_b = np.zeros((rh, rw), dtype=np.float32)

np.add.at(img_r, (py, px), r_vals)
np.add.at(img_g, (py, px), g_vals)
np.add.at(img_b, (py, px), b_vals)

# Same Gaussian smoothing
sigma = max(rw, rh) * SIGMA_SCALE
img_r = gaussian_filter(img_r, sigma=sigma)
img_g = gaussian_filter(img_g, sigma=sigma)
img_b = gaussian_filter(img_b, sigma=sigma)

# Resize to final image size
img_r_resized = cv2.resize(img_r, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)
img_g_resized = cv2.resize(img_g, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)
img_b_resized = cv2.resize(img_b, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)


# Stack and convert to RGB
rgb_image = np.stack([img_r_resized, img_g_resized, img_b_resized], axis=-1)
rgb_image = (rgb_image * 255).astype(np.uint8)

# Clean white background
background_mask = (img_r_resized == 0) & (img_g_resized == 0) & (img_b_resized == 0)
rgb_image[background_mask] = 255

max_intensity = max(img_r_resized.max(), img_g_resized.max(), img_b_resized.max())
low_thresh = max_intensity * 0.01
low_mask = (img_r_resized < low_thresh) & (img_g_resized < low_thresh) & (img_b_resized < low_thresh)
rgb_image[low_mask] = 255

# Save the true color H&E image
output_true_rgb = BASE_FOLDER / "HE_True_RGB_Color.png"
Image.fromarray(rgb_image).save(output_true_rgb)
print(f"Saved true full-color RGB H&E image: {output_true_rgb}")

####################################
# save csv dataset
df_pep.to_csv(BASE_FOLDER / "peptide_df_aligned_to_HE_all_cores.csv", index=False)
