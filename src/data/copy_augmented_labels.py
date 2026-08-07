import pandas as pd
from pathlib import Path
import shutil

FOG_LOG = Path("data/augmentation_log.csv")
LOWLIGHT_LOG = Path("data/lowlight_augmentation_log.csv")
FOUND_COL = "found"
ORIGINAL_LABELS_DIR = Path("data/labels")
FILENAME_COL = "filename"
OUTPUT_PATH = "output_path"


def build_label_pairs(found_df):
    pairs = []
    for _, row in found_df.iterrows():
        original_stem = Path(row[FILENAME_COL]).stem #Just the file name
        original_label = ORIGINAL_LABELS_DIR/f"{original_stem}.txt"
        new_label = Path(row[OUTPUT_PATH]).with_suffix(".txt")
        pairs.append((original_label,new_label))
    return pairs

fog_df = pd.read_csv(FOG_LOG)
lowlight_df = pd.read_csv(LOWLIGHT_LOG)

fog_found = fog_df[fog_df[FOUND_COL]==True]
lowlight_found = lowlight_df[lowlight_df[FOUND_COL] == True]


print(f"Fog Log: {len(fog_df)} rows total, {len(fog_found)} found = True")
print(f"Low light log:{len(lowlight_df)} rows total, {len(lowlight_found)} found = True")


fog_pairs = build_label_pairs(fog_found)
lowlight_pairs = build_label_pairs(lowlight_found)

missing = []

for original_label, new_label in (fog_pairs + lowlight_pairs):
    if not original_label.exists():
        missing.append(original_label)

print(f'Fog: {len(fog_pairs)} pairs built')
print(f'Low-light: {len(lowlight_pairs)} pairs built')
print(f'Missing Original labels: {len(missing)}')

print("\nsample pairs:")
for orig, new in fog_pairs[:2]:
    print(f'{orig} -> {new}')

print('\n missing labels:')
for m in missing:
    print(f'{m}')

all_pairs = fog_pairs + lowlight_pairs

print(f"\n About to copy{len(all_pairs)} label files")
print(f'Fog:{len(fog_pairs)}')
print(f'Low-light:{len(lowlight_pairs)}')

copied = 0
errors = []

for origlabel, newlabel in all_pairs:
    try:
        shutil.copy2(origlabel,newlabel)
        copied += 1

        if copied % 500 == 0:
            print(f'Copied {copied}/{len(all_pairs)}...')
    except Exception as e:
        errors.append((origlabel,newlabel, str(e)))
print(f'Done. Copied:{copied}, Errors: {len(errors)}')
if errors:
    print('\n Fisrt few errors:')
    for orig, new, err in errors[:5]:
        print(f'{orig} -> {new} : {err}')