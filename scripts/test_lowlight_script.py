import cv2
from src.calibration.low_light_severity import LowLightSeverity
from src.augmentation.lowlight_augmentation import (
    compute_lowlight_threshold,
)
from scripts.run_lowlight_augmentation import process_row, RAW_IMAGE_DIR
import os
import pandas as pd

low_light_severity = LowLightSeverity()
p33, p66 = compute_lowlight_threshold()

df = pd.read_csv('data/severity_scores_v2.csv')
row = df[df['filename'] == 'AM_Google_10.png'].iloc[0]

result = process_row(row, p33=p33, p66=p66)
print(f"row score: {row['low_light_severity']}")
print(f"p33={p33}   p66={p66}")

result = process_row(row, p33=p33, p66=p66)
print(result)