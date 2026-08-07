import pandas as pd

COL_HAZE_BIN = 'bin'
COL_LOW_LIGHT_BIN = 'low_light_bin'
CSV_PATH = 'data/severity_scores_v2.csv'

df = pd.read_csv(CSV_PATH)

blind_spot_count = ((df[COL_HAZE_BIN]=='low') & (df[COL_LOW_LIGHT_BIN]=='high')).sum()
#low haze but scarcity of light is high
total = len(df)
pct = 100 * blind_spot_count/total
print(f'Low haze images but scarcity of light flag as severe:{blind_spot_count}/{total} ({pct:.1f}%)')