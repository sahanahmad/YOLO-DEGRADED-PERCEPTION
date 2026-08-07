import pandas as pd
import cv2
from src.augmentation.fog_augmentation import(
    compute_threshold,compute_ceiling,decide_bucket,apply_full_fog,
    apply_fog_in_range,save_augmented_image
)
import multiprocessing
from functools import partial
CSV_FILE = 'data/severity_scores.csv'
LOG_FILE = 'data/augmentation_log.csv'
IMAGE_DIR = 'data/raw/RTTS/JPEGImages'
FILENAME = 'filename'
COL_SEVERITY = 'severity'
NUM_WORKERS = 4

def process_row(row, p33,p66,ceiling):
    filename = row[FILENAME]
    severity = row[COL_SEVERITY]
    bucket = decide_bucket(severity,p33=p33,p66=p66)
    img = cv2.imread(f'{IMAGE_DIR}/{filename}')

    log_rows = []
    if bucket == 'augment_full':
        results = apply_full_fog(image_bgr=img, p33=p33, p66=p66, ceiling=ceiling)
        for tier, (aug_img, aug_severity, found) in results.items():
            output_path = save_augmented_image(aug_img,filename,tier) if found else None
            log_rows.append({'filename':filename, 'bucket':bucket, 'tier':tier,
                            'found':found, 'achieved_severity':aug_severity, 
                            'output_path':output_path}) 

    elif bucket == 'augment_medium':
        aug_img, aug_severity, found = apply_fog_in_range(image_bgr=img, lower_bound=0, upper_bound=ceiling)
        output_path = save_augmented_image(aug_img,filename,'medium') if found else None
        log_rows.append({'filename':filename, 'bucket':bucket, 'tier':'medium',
                            'found':found, 'achieved_severity':aug_severity, 
                            'output_path':output_path}) 
    else:    
        log_rows.append({
        'filename': filename, 'bucket': bucket, 'tier': None,
        'found': None, 'achieved_severity': None, 'output_path': None
        })

    return log_rows


if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE)
    #df = df.head(100) #Temporary- remove this once small test passes.
    p33, p66 = compute_threshold()
    ceiling = compute_ceiling()

    records = df.to_dict('records')
    worker_fn = partial(process_row, p33=p33, p66=p66, ceiling=ceiling)
    #partial's whole job is lock in some arguments, leave others open for later.

    log_rows = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap(worker_fn, records)):
            log_rows.extend(result)
            if (i+1)%500 == 0:
                print(f'Processed {i+1}/{len(df)} Images')
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_FILE, index=False)
    print('=== Summary ===')
    print(log_df.drop_duplicates(subset='filename')['bucket'].value_counts())
    print('Tier failure counts:')
    print(log_df[log_df['found'] == False]['tier'].value_counts())