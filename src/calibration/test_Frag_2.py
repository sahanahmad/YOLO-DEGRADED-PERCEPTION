from low_light_severity import LowLightSeverity
import cv2
import glob

scores = LowLightSeverity()

image_path = {'GSGL_Google_120.png': 'data/raw/RTTS/JPEGImages/GSGL_Google_120.png',
    'FogDr_Bing_569.png': 'data/raw/RTTS/JPEGImages/FogDr_Bing_569.png'}

for name, path in image_path.items():
    img = cv2.imread(path)
    point = scores.score(image_bgr=img)
    print(f'{name}: {point:.2f}')
    print(f'{name}:{scores.severity_bin(point)}')

image_path_glob = glob.glob('data/raw/RTTS/JPEGImages/*.png')
print(f'Found {len(image_path_glob)} images')

