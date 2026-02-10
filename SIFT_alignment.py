#SIFT algorithm alignment
#this script get ref images and give wrap vector and overlay image

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Paths to load images
#ref image Glycan
REF_FIXED = "S:/img_vec/SIFT_based/261/REF_Glycan_m.z.sum_5.png"  #ref_Ion_Image_261_PNGase
#ref image Peptide
REF_MOVING = "S:/img_vec/SIFT_based/261/REF_Peptide_m.z.sum_5.png" #ref_Ion_Image_261_Trypsin

OUTPUT_REF = "S:/img_vec/SIFT_based/261/2"  #Out put folder
os.makedirs(OUTPUT_REF, exist_ok=True)

#check to ensure be loaded correct
print(os.path.exists(REF_FIXED))      # Should be True
print(os.path.exists(REF_MOVING))     # Should be True

# Image setting Parameters
PERCENTILE = 70
GAMMA = 0.6
SMOOTH_K = (5,5)
CMAP_FIXED = "Blues"  #glycan image color
CMAP_MOVING = "Reds"  #peptide image color
MAX_OPACITY = 0.9

# Functions
#Load image and convert to gray
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
        raise ValueError("Expected 3 or 4 channels")

def composite(bg, fg, alpha):
    return fg*alpha[...,None] + bg*(1.0-alpha[...,None])

#### SIFT
def align_sift(fixed_gray, moving_gray, ratio_thresh=0.75, min_match_count=10):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(fixed_gray, None)
    kp2, des2 = sift.detectAndCompute(moving_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    print(f" Found {len(good)} good matches and number of matched key points")

    if len(good) > min_match_count:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        warp_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if warp_matrix is None:
            print("SIFT alignment failed !!!")
            warp_matrix = np.eye(2,3,dtype=np.float32)
    else:
        print(f"Not enough matches ({len(good)}) !!!")
        warp_matrix = np.eye(2,3,dtype=np.float32)

    warped = cv2.warpAffine(
        moving_gray,
        warp_matrix,
        (fixed_gray.shape[1], fixed_gray.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return warped, warp_matrix, len(good)

def process_image_pair(moving_path, fixed_path, out_folder, warp_matrix=None):
    moving = load_gray(moving_path)
    fixed = load_gray(fixed_path)
    if moving.shape != fixed.shape:
        moving = cv2.resize(moving, (fixed.shape[1], fixed.shape[0]), interpolation=cv2.INTER_AREA)

    fname = os.path.splitext(os.path.basename(moving_path))[0]

    # Save raw moving vector
    raw_vec_path = os.path.join(out_folder, f"{fname}_raw_vector.csv")
    np.savetxt(raw_vec_path, moving.flatten(), delimiter=",")


    # Align or apply warp
    if warp_matrix is None:
        warped_moving, warp_matrix, num_matches = align_sift(fixed, moving)
        print(f" Aligned {fname}, SIFT good matches={num_matches}")
    else:
        warped_moving = cv2.warpAffine(moving, warp_matrix, (fixed.shape[1], fixed.shape[0]),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT)


    fixed_f = fixed.astype(np.float32)/255.0
    moving_f = warped_moving.astype(np.float32)/255.0

    s_fixed = compute_strength(fixed_f)
    s_moving = compute_strength(moving_f)

    alpha_fixed = np.clip(s_fixed*MAX_OPACITY,0,1)
    alpha_moving = np.clip(s_moving*MAX_OPACITY,0,1)

    col_fixed = apply_colormap(s_fixed, CMAP_FIXED)
    col_moving = apply_colormap(s_moving, CMAP_MOVING)

    # overlay image
    canvas = composite(np.stack([fixed_f]*3,-1), col_fixed, alpha_fixed)
    canvas = composite(canvas, col_moving, alpha_moving)
    canvas = np.clip(canvas,0,1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S") #timestamp string for namings

    # Save images
    save_png(os.path.join(out_folder,f"{fname}_fixed_blue_{ts}.png"), np.dstack([col_fixed, alpha_fixed]))
    save_png(os.path.join(out_folder,f"{fname}_moving_red_{ts}.png"), np.dstack([col_moving, alpha_moving]))
    save_png(os.path.join(out_folder,f"{fname}_overlay_{ts}.png"), canvas)
    # Save final moving vector
    final_vec_path = os.path.join(out_folder, f"{fname}_final_vector.csv")
    np.savetxt(final_vec_path, (col_moving*alpha_moving[...,None]).reshape(-1,3), delimiter=",")

    return warp_matrix

 #run to process images
warp_matrix = process_image_pair(REF_MOVING, REF_FIXED, OUTPUT_REF)
np.savetxt(os.path.join(OUTPUT_REF, "ref_warp_matrix.csv"), warp_matrix, delimiter=",")

print("All images and tables saved")
