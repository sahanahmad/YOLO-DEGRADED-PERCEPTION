import pandas as pd
from src.data.voc_to_yolo_v2 import dark_channel
import albumentations as A
import os
import cv2

CSV_FILE = 'data/severity_scores.csv'
AUGMENTED_IMAGE_DIRECTORY = 'data/processed/fog_augmented'
SEVERITY_COL = 'severity'
def compute_threshold():
    df = pd.read_csv(CSV_FILE)
    p33 = df[SEVERITY_COL].quantile(0.33)
    p66 = df[SEVERITY_COL].quantile(0.66)

    return p33,p66

def decide_bucket(dc_value, p33, p66):
    if dc_value < p33:
        return 'augment_full'
    elif dc_value < p66:
        return 'augment_medium'
    return 'no_augment'

def compute_ceiling():
    df = pd.read_csv(CSV_FILE)
    return df[SEVERITY_COL].quantile(.90)

def apply_fog_in_range(image_bgr, lower_bound, upper_bound, alpha_coef=0.12):
    aug_fog_strength =[(0.02,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),
                  (0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,0.98),(0.98,1.0)]

    best_image = None
    best_severity = None

    for low, high in aug_fog_strength:
        transform = A.RandomFog(fog_coef_range=(low,high), alpha_coef=alpha_coef,p=1.0)
        augmented = transform(image=image_bgr)['image']
        #image specific keywords other option mask,bboxes,keypoints
        severity = dark_channel(augmented)

        if lower_bound <= severity < upper_bound:
            best_image = augmented
            best_severity = severity
    found = best_image is not None
    return best_image, best_severity, found

def save_augmented_image(image_bgr, original_filename, suffix):
    os.makedirs(AUGMENTED_IMAGE_DIRECTORY, exist_ok= True)
    filename_no_ext = os.path.splitext(original_filename)[0]
    output_filename = f'{filename_no_ext}_{suffix}.png'
    output_path = os.path.join(AUGMENTED_IMAGE_DIRECTORY, output_filename)
    cv2.imwrite(output_path,image_bgr)
    return output_path

def apply_full_fog(image_bgr, p33, p66, ceiling, alpha_coef=0.12):
    results = {}
    for tier, (lower, upper) in [('low', (0,p33)), ('medium', (p33,p66)), ('high',(p66,ceiling))]:
        img, severity, found = apply_fog_in_range(image_bgr=image_bgr,lower_bound=lower, upper_bound=upper,alpha_coef=alpha_coef)
        if not found:
            print(f'WARNING: could not find a fog strenght landing in the {tier} range[{lower:.2f}, {upper:.2f}]')
        results[tier] = (img, severity,found)

    return results
