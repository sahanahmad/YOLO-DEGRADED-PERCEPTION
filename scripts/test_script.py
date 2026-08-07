import cv2
from src.calibration.low_light_severity import LowLightSeverity
from src.augmentation.lowlight_augmentation import apply_medium_darkness
import numpy as np
import albumentations as A

img = cv2.imread("data/raw/RTTS/JPEGImages/AM_Google_10.png")
low_light_severity = LowLightSeverity()
before_score = low_light_severity.score(img)
print(f"before:{before_score:.2f}")

candidates = np.round(np.arange(-0.1,-1.0,-0.1),1)
#best_img, best_score, found = apply_medium_darkness(image_bgr=img,lower_bound=134.24, upper_bound=157.51, low_light_severity=low_light_severity)
#print(f'found:{found} best score:{best_score}')

for b in candidates:
    transform = A.RandomBrightnessContrast(brightness_limit=(b,b),contrast_limit=0,p=1)
    darkend = transform(image=img)["image"]
    score = low_light_severity.score(darkend)
    print(f"b={b:.2f} score={score:.2f}")
