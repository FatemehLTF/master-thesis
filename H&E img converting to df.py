### used to convert H&E color image in to dataset containing gray signal and RGB signals
import numpy as np
import cv2
import pandas as pd

# Load H&E image

def load_slide(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Remove alpha channel if present
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


# Preprocess for tissue/core detection
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphology to clean up and connect tissue regions
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    return thresh

# Find cores

def find_cores(thresh, min_area=500): #min-area can be changed
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return contours

# Extract grayscale intensity + RGB pixels
def extract_pixels(slide_img, contours):
    h, w = slide_img.shape[:2]
    pixels = []
    core_id = 1
    for cnt in contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
        ys, xs = np.where(mask == 255)
        if len(xs) == 0:
            continue
        for pid, (y, x) in enumerate(zip(ys, xs)):
            intensity = int(cv2.cvtColor(slide_img, cv2.COLOR_RGB2GRAY)[y, x])
            r, g, b = slide_img[y, x]
            pixels.append({
                "Core": f"Core_{core_id}",
                "Pixel_ID": pid,
                "x": int(x),
                "y": int(y),
                "Intensity": intensity,
                "R": int(r),
                "G": int(g),
                "B": int(b)
            })
        core_id += 1
    return pd.DataFrame(pixels), core_id - 1


# Reconstruct grayscale image
def reconstruct_image(pixels_df, shape):
    gray = np.zeros(shape[:2], dtype=np.uint8)
    for _, row in pixels_df.iterrows():
        gray[int(row["y"]), int(row["x"])] = int(row["Intensity"])
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# Label cores with numbers (used for annotation in TMA)
def label_cores(img, pixels_df):
    labeled = img.copy()
    for core, group in pixels_df.groupby("Core"):
        cx = int(group["x"].mean())
        cy = int(group["y"].mean())
        core_num = core.split("_")[1]
        cv2.putText(
            labeled, core_num, (cx - 20, cy + 10),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA  # Red text
        )
    return labeled

# Save image
def save_image(img, path):
    annotated = img.copy()
    cv2.putText(
        annotated, " ", (30, annotated.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA
    )
    cv2.imwrite(path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

### main code:
slide_path = r"S:\H&E_wholemount\H&E.png"
output_csv = r"S:\H&E_wholemount\H&E_df.csv"
output_img = r"S:\H&E_wholemount\H&E_labeled.png"

# Load image
slide_img = load_slide(slide_path)

# Detect cores
thresh = preprocess_image(slide_img)
contours = find_cores(thresh, min_area=500)  # Adjust min_area

# Extract intensity + RGB pixels
pixels_df, num_cores = extract_pixels(slide_img, contours)

# Save CSV
pixels_df.to_csv(output_csv, index=False)

#  saveimage
reconstructed = reconstruct_image(pixels_df, slide_img.shape)
labeled = label_cores(reconstructed, pixels_df)
save_image(labeled, output_img)