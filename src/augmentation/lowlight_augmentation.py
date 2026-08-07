import numpy as np
import albumentations as A
import os
import cv2
import pandas as pd

AUGMENTED_IMAGE_DIRECTORY = 'data/processed/lowlight_augmented'
CSV_FILE = 'data/severity_scores_v2.csv'
LOW_LIGHT_COL = 'low_light_severity'


def decide_lowlight_bucket(score,p33,p66):
    if score >= p66:
        return 'augment_medium'
    return 'no_augment'

def apply_medium_darkness(image_bgr, lower_bound, upper_bound, low_light_severity):
    candidates = np.round(np.arange(-0.1,-1.0,-0.1), 1)
    for b in candidates:
        transform = A.RandomBrightnessContrast(brightness_limit=(b,b),contrast_limit=0, p=1)
        candidate_img = transform(image=image_bgr)['image']
        candidate_score = low_light_severity.score(candidate_img)

        if lower_bound <= candidate_score < upper_bound:
            return candidate_img, candidate_score, True

    return None, None, False

def save_lowlight_augmented_image(image_bgr, original_filename, suffix):
    os.makedirs(AUGMENTED_IMAGE_DIRECTORY, exist_ok= True)
    filename_no_ext = os.path.splitext(original_filename)[0]
    output_filename = f'{filename_no_ext}_{suffix}.png'
    output_path = os.path.join(AUGMENTED_IMAGE_DIRECTORY,output_filename)
    cv2.imwrite(output_path,image_bgr)
    return output_path

def compute_lowlight_threshold():
    df = pd.read_csv(CSV_FILE)
    p33 = df[LOW_LIGHT_COL].quantile(0.33)
    p66 = df[LOW_LIGHT_COL].quantile(0.66)
    return p33, p66
