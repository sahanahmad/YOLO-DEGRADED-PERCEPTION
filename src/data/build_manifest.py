import pandas as pd
from pathlib import Path

IMG_FOLDER = Path("data/raw/RTTS/JPEGImages")
LABELS_FOLDER = Path("data/labels")
FOG_LOG = Path("data/augmentation_log.csv")
LOWLIGHT_LOG = Path("data/lowlight_augmentation_log.csv")
MANIFEST_CSV = Path("data/manifest.csv")

orignial_rows = []

for img_path in sorted(IMG_FOLDER.glob("*.png")):
    stem = img_path.stem
    label_path = LABELS_FOLDER / f'{stem}.txt'
    orignial_rows.append({
        "image_path": str(img_path),
        "label_path": str(label_path),
        "source_stem": stem,
        "augmentation_type": "original"
    })

#print(f'Original rows build:{len(orignial_rows)}')
#print(orignial_rows[0])

def build_augmented_rows(log_path, aug_type):
    df = pd.read_csv(log_path)
    found_df = df[df["found"] == True]

    rows = []
    for _, row in found_df.iterrows():
        source_stem = Path(row["filename"]).stem
        image_path = Path(row["output_path"])
        label_path = image_path.with_suffix(".txt")

        rows.append({
            "image_path": str(image_path),
            "label_path": str(label_path),
            "source_stem": source_stem,
            "augmentation_type": aug_type
        })
    return rows

fog_rows = build_augmented_rows(FOG_LOG, 'fog')
lowlight_rows = build_augmented_rows(LOWLIGHT_LOG, 'lowlight')

#print(f'Fog rows: {len(fog_rows)}')
#print(f'Low-light rows:{len(lowlight_rows)}')
#print(fog_rows[0])
#print(lowlight_rows[0])


all_rows = orignial_rows + fog_rows + lowlight_rows
manifest_df = pd.DataFrame(all_rows)
print(f'\n Total manifest rows:{len(manifest_df)}')

missing_images = [p for p in manifest_df["image_path"] if not Path(p).exists()]
missing_labels = [p for p in manifest_df["label_path"] if not Path(p).exists()]

print(f'Missing Images: {len(missing_images)}')
print(f'Missing labels: {len(missing_labels)}')

group_counts = manifest_df.groupby("source_stem").size()
print(group_counts.value_counts().sort_index())
print(f'\n Stems with 1 row(original only): {(group_counts == 1).sum()}')
print(f'Stems with 2 rows:{(group_counts == 2).sum()}')
print(f'Stems with 3 rows:{(group_counts == 3).sum()}')

manifest_df.to_csv(MANIFEST_CSV, index= False)
print(f'\nWritten:{MANIFEST_CSV}')