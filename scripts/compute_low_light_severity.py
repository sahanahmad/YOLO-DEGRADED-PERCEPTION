import glob
import cv2
from src.calibration.low_light_severity import LowLightSeverity
import pandas as pd
import os

COL_FILENAME = 'filename'
COL_LOW_LIGHT_SEVERITY = 'low_light_severity'
COL_LOW_LIGHT_BIN = 'low_light_bin'
IMAGE_GLOB = glob.glob('data/raw/RTTS/JPEGImages/*.png')
CSV_WITH_HAZE_SEVERITY = 'data/severity_scores.csv'
CSV_WITH_LOWLIGHT_SEVERITY = 'data/severity_scores_v2.csv'

severity = LowLightSeverity()
luminance_scores = []

for path in IMAGE_GLOB:
    img = cv2.imread(path)
    score = severity.score(image_bgr=img)
    luminance_scores.append((path, score))

print(f'Scored {len(luminance_scores)} Images')
for filename, score in luminance_scores[:5]:
    print(f'{filename}: {score:.2f}')
# NEW — build the lookup dict, converting full paths to bare filenames
luminance_by_filename = {}
for path,score in luminance_scores:
    filename = os.path.basename(path)
    luminance_by_filename[filename]=score
# join onto the real CSV
df = pd.read_csv(CSV_WITH_HAZE_SEVERITY)
df[COL_LOW_LIGHT_SEVERITY] = df[COL_FILENAME].map(luminance_by_filename)
#df.to_csv(CSV_WITH_LOWLIGHT_SEVERITY, index=False)
print(df.head())

missing = df['low_light_severity'].isna().sum()
print(f'Rows with missing low_light_severity:{missing}')

luminance_values = df[COL_LOW_LIGHT_SEVERITY].tolist()
severity.fit(luminance_values=luminance_values)
print(f'p33: {severity.p33:.2f}')
print(f'p66: {severity.p66:.2f}')


df[COL_LOW_LIGHT_BIN] = df[COL_LOW_LIGHT_SEVERITY].apply(severity.severity_bin)
df.to_csv(CSV_WITH_LOWLIGHT_SEVERITY, index= False)
print(df.head())
print(df[COL_LOW_LIGHT_BIN].value_counts())
missing = df[COL_LOW_LIGHT_BIN].isna().sum()
print(f'Rows with missing low light severity:{missing}')