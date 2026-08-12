import pandas as pd
import numpy as np
from pathlib import Path

MANIFEST_CSV = Path("data/manifest.csv")
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
manifest_df = pd.read_csv(MANIFEST_CSV)
stem_to_split = {}
FINAL_MANIFEST_CSV = Path("data/manifest_with_splits.csv")

stems = manifest_df["source_stem"].unique()
print(f"Total unique stems:{len(stems)}")

rng = np.random.default_rng(SEED)
shuffled_stems = stems.copy()
rng.shuffle(shuffled_stems)

n_train = int(len(shuffled_stems)*TRAIN_FRAC)
n_val = int(len(shuffled_stems)*VAL_FRAC)

train_stems = shuffled_stems[:n_train]
val_stems = shuffled_stems[n_train:n_train+n_val]
test_stems = shuffled_stems[n_train+n_val:]

print(f'Train Stems:{len(train_stems)}')
print(f'Val Stems: {len(val_stems)}')
print(f'Test stems:{len(test_stems)}')


for stem in train_stems:
    stem_to_split[stem] = 'train'
for stem in val_stems:
    stem_to_split[stem] = 'val'
for stem in test_stems:
    stem_to_split[stem] = 'test'

manifest_df['split'] = manifest_df['source_stem'].map(stem_to_split)
print("Split Counts (rows):")
print(manifest_df['split'].value_counts())


unassigned = manifest_df['split'].isna().sum()
print(f'\nRows with no split assigned:{unassigned}')

sample_stem = "AM_Bing_211"
print(f'\n{sample_stem}rows:')
print(manifest_df[manifest_df["source_stem"] == sample_stem][["image_path", "augmentation_type", "split"]])

manifest_df.to_csv(FINAL_MANIFEST_CSV,index=False)
print(f'\nWritten:{FINAL_MANIFEST_CSV}')