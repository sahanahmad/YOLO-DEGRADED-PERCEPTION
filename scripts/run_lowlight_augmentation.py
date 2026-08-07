import cv2
import pandas as pd
import multiprocessing
from functools import partial
from src.calibration.low_light_severity import LowLightSeverity
from src.augmentation.lowlight_augmentation import(
    decide_lowlight_bucket,
    apply_medium_darkness,
    save_lowlight_augmented_image,
    compute_lowlight_threshold
)

IMAGE_DIR = 'data/raw/RTTS/JPEGImages'
FILENAME = 'filename'
COL_LOW_LIGHT_SEVERITY = 'low_light_severity'
CSV_FILE = 'data/severity_scores_v2.csv'
LOG_FILE = 'data/lowlight_augmentation_log.csv'
NUM_WORKERS = 4

def process_row(row, p33, p66):
    low_light_severity = LowLightSeverity()
    filename = row[FILENAME]
    score = row[COL_LOW_LIGHT_SEVERITY]
    bucket = decide_lowlight_bucket(score= score, p33=p33,p66=p66)
    img = cv2.imread(f'{IMAGE_DIR}/{filename}')

    log_rows = []

    if bucket == 'augment_medium':
        darkend_image, acheived_score, found = apply_medium_darkness(
            image_bgr=img,lower_bound=p33,upper_bound=p66, low_light_severity=low_light_severity
        )
        output_path = None
        if found:
            output_path = save_lowlight_augmented_image(image_bgr=darkend_image,original_filename=filename,suffix='medium_dark')

        log_rows.append({'filename': filename, 'bucket': bucket, 'tier': 'medium_dark',
                          'found': found, 'achieved_score': acheived_score,
                          'output_path': output_path})
    else:
        log_rows.append({
            'filename': filename, 'bucket': bucket, 'tier': None,
            'found': None, 'achieved_score': None, 'output_path': None
        })
    return log_rows

if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE)
    #df = df.head(20) # Temporary-remove this once small test passes
    p33, p66 = compute_lowlight_threshold()

    records = df.to_dict('records')
    worker_fn = partial(process_row, p33=p33, p66=p66)

    log_rows = []

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap(worker_fn,records)):
            log_rows.extend(result)
            if (i+1) % 500 == 0:
                print(f'Processed{i+1}/{len(df)} Images')
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_FILE, index=False)
    print('====Summary===')
    print(log_df.drop_duplicates(subset='filename')['bucket'].value_counts())
    print('Found counts (augment medium only)')
    print(log_df[log_df['bucket']== 'augment_medium']['found'].value_counts())
