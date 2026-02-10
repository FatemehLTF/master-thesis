#ECC alginment:
#Get 2 ref imegse of glycan and peptide and aligned them

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Paths and loading images
#ref image Glycan
REF_FIXED = "S:/img_vec/ECC_based/261/REF_Glycan_m.z.sum_5.png"  #ref_Ion_Image_261_PNGase
#ref image Peptide
REF_MOVING = "S:/img_vec/ECC_based/261/REF_Peptide_m.z.sum_5.png" #ref_Ion_Image_261_Trypsin
OUTPUT_REF = "S:/img_vec/ECC_based/261"
os.makedirs(OUTPUT_REF, exist_ok=True)

#check
print(os.path.exists(REF_FIXED))      # Should be True
print(os.path.exists(REF_MOVING))     # Should be True

# Image Parameters
PERCENTILE = 70
GAMMA = 0.6
SMOOTH_K = (5,5)
CMAP_FIXED = "Blues"  # glycan images
CMAP_MOVING = "Reds"  # peptide images
MAX_OPACITY = 0.9

# Functions
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Couldn't read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def compute_strength(img_f):
    raw = 1.0 - img_f
    thr = np.percentile(raw, PERCENTILE)
    raw_masked = np.where(raw >= thr, raw, 0.0)
    if raw_masked.max() > 0:
        raw_masked /= raw_masked.max()
    if GAMMA != 1.0:
        raw_masked = np.power(raw_masked, GAMMA)
        if raw_masked.max() > 0:
            raw_masked /= raw_masked.max()
    if SMOOTH_K:
        raw_masked = cv2.GaussianBlur(raw_masked.astype(np.float32), SMOOTH_K, 0)
        if raw_masked.max() > 0:
            raw_masked /= raw_masked.max()
    return np.clip(raw_masked, 0.0, 1.0)

def apply_colormap(strength, cmap_name):
    return plt.get_cmap(cmap_name)(strength)[..., :3].astype(np.float32)

def save_png(path, arr):
    arr8 = (np.clip(arr,0,1)*255).astype(np.uint8)
    if arr8.shape[-1]==3:
        cv2.imwrite(path, cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR))
    elif arr8.shape[-1]==4:
        bgra = arr8[..., [2,1,0,3]]
        cv2.imwrite(path,bgra)
    else:
        raise ValueError("Channels")

def composite(bg, fg, alpha):
    return fg*alpha[...,None] + bg*(1.0-alpha[...,None])

def align_ecc(fixed_gray, moving_gray):
    warp_matrix = np.eye(2,3,dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    try:
        cc, warp_matrix = cv2.findTransformECC(fixed_gray.astype(np.float32)/255.0,
                                               moving_gray.astype(np.float32)/255.0,
                                               warp_matrix,
                                               cv2.MOTION_AFFINE,
                                               criteria)
        warped = cv2.warpAffine(moving_gray, warp_matrix, (fixed_gray.shape[1], fixed_gray.shape[0]),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT)
        return warped, warp_matrix, cc
    except cv2.error as e:
        print(f" ECC failed: {e}")
        return moving_gray.copy(), warp_matrix, 0.0

def process_image_pair(moving_path, fixed_path, out_folder, warp_matrix=None):
    moving = load_gray(moving_path)
    fixed = load_gray(fixed_path)
    if moving.shape != fixed.shape:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_AREA)

    fname = os.path.splitext(os.path.basename(moving_path))[0]

    # Save raw moving vector
    raw_vec_path = os.path.join(out_folder, f"{fname}_raw_vector.csv")
    np.savetxt(raw_vec_path, moving.flatten(), delimiter=",")
    print(f"{raw_vec_path}")

    # Align or apply warp
    if warp_matrix is None:
        warped_moving, warp_matrix, cc = align_ecc(fixed, moving)
        print(f"Aligned {fname}, ECC={cc:.4f}")
    else:
        warped_moving = cv2.warpAffine(moving, warp_matrix, (fixed.shape[1], fixed.shape[0]),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT)
        print(f"Applied reference warp to {fname}")

    fixed_f = fixed.astype(np.float32)/255.0
    moving_f = warped_moving.astype(np.float32)/255.0

    s_fixed = compute_strength(fixed_f)
    s_moving = compute_strength(moving_f)

    alpha_fixed = np.clip(s_fixed*MAX_OPACITY,0,1)
    alpha_moving = np.clip(s_moving*MAX_OPACITY,0,1)

    col_fixed = apply_colormap(s_fixed, CMAP_FIXED)
    col_moving = apply_colormap(s_moving, CMAP_MOVING)

    # Overlay
    canvas = composite(np.stack([fixed_f]*3,-1), col_fixed, alpha_fixed)
    canvas = composite(canvas, col_moving, alpha_moving)
    canvas = np.clip(canvas,0,1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S") #automatic naming

    # Save images
    save_png(os.path.join(out_folder,f"{fname}_fixed_blue_{ts}.png"), np.dstack([col_fixed, alpha_fixed]))
    save_png(os.path.join(out_folder,f"{fname}_moving_red_{ts}.png"), np.dstack([col_moving, alpha_moving]))
    save_png(os.path.join(out_folder,f"{fname}_overlay_{ts}.png"), canvas)

    # Save final moving vector
    final_vec_path = os.path.join(out_folder, f"{fname}_final_vector.csv")
    np.savetxt(final_vec_path, (col_moving*alpha_moving[...,None]).reshape(-1,3), delimiter=",")
    print(f"{final_vec_path}")

    return warp_matrix

# codes main part : Run code
warp_matrix = process_image_pair(REF_MOVING, REF_FIXED, OUTPUT_REF)
np.savetxt(os.path.join(OUTPUT_REF, "ref_warp_matrix.csv"), warp_matrix, delimiter=",")
print(" All images and excel saved ")
