import os
import pandas as pd
from tqdm import tqdm

def get_lifecycle_staging(dir):
    sample_packets = sorted([f for f in os.listdir(dir) if f.startswith('acc')])
    labels = {}
    n = len(sample_packets)
    for stages in tqdm(range(2,8), desc = f"Lifecyle Staging for {dir}"):
        labels[stages] = []
        start_idx = 0
        for i in range(1,stages+1):
            stage_len = int(pow(0.5, i)*n)
            rem_len = n - start_idx
            if stage_len == 0 or rem_len == 0:
                break
            start_idx += min(stage_len, rem_len)
            labels[stages] += [i-1]*(min(stage_len, rem_len))
        if start_idx < n-1:
            labels[stages] += [stages-1]*(n-start_idx)

    # storing as dataframe
    df = pd.DataFrame(columns=['dir' , 'file_name']+[f"stage_{i}" for i in range(2,8)])
    for i , file in enumerate(sample_packets):
        file_name = file.split('.')[0]
        df.loc[len(df)] = [dir] + [file_name] + [labels[stages][i] for stages in range(2,8)]

    return df

def main(dir, save_dir):
    final_df = None
    for root, dirs, files in os.walk(dir):
        for d in dirs:
            if 'fft' in d:
                df = get_lifecycle_staging(os.path.join(root, d))
                if final_df is None:
                    final_df = df
                else:
                    final_df = pd.concat([final_df, df], axis=0)

    # shuffle the dataframe
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    final_df.to_csv(f'{save_dir}/life_cycle_staging_{os.path.basename(dir)}.csv', index=False)

if __name__ == "__main__":
    SAVE_DIR = 'data/ieee-RUL'
    for dir in ['data/ieee-RUL/Full_Test_Set', 'data/ieee-RUL/Learning_set', 'data/ieee-RUL/Test_set']:
        main(dir, SAVE_DIR)


