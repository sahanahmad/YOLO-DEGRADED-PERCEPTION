from src.augmentation.fog_augmentation import compute_threshold, decide_bucket

p33, p66 = compute_threshold()
print(f'p33: {p33:.2f}')
print(f'p66: {p66:.2f}')

print(decide_bucket(42.63, p33, p66))    # AM_Google_10.png  -> expect augment_full
print(decide_bucket(107.75, p33, p66))   # AM_Google_617.png -> expect middle bucket
print(decide_bucket(197.18, p33, p66))   # AM_Google_707.png -> expect no_augment