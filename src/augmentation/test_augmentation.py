from src.augmentation.fog_augmentation import compute_threshold, decide_bucket, compute_ceiling,apply_full_fog
import cv2
from src.data.voc_to_yolo_v2 import dark_channel
p33, p66 = compute_threshold()
ceiling = compute_ceiling()
img = cv2.imread('data/raw/RTTS/JPEGImages/AM_Google_10.png')
print(f'Input Severity:{dark_channel(image_bgr=img):.2f}')

resuls= apply_full_fog(image_bgr=img,p33=p33,p66=p66,ceiling=ceiling)

for tier, (aug_img, severity, found) in resuls.items():
    status = f'{severity:.2f}' if found else 'FAILED'
    print(f'{tier}:{status}')
