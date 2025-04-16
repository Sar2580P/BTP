import os
import glob
import numpy as np
import pandas as pd
from scipy.fft import fft
from pydantic import BaseModel, DirectoryPath, StrictStr
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import logging
import json

class CSVProcessor(BaseModel):
    input_dir: DirectoryPath
    save_dir: StrictStr
    save_plot: StrictStr = "pics"
    error_log: StrictStr = "error.json"

    def __init__(self, **data):
        super().__init__(**data)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_plot,exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename=self.error_log, level=logging.ERROR, format='%(asctime)s %(message)s')

    def log_error(self, message: str):
        with open(self.error_log, 'a') as f:
            json.dump({"error": message}, f)
            f.write('\n')

    def merge_last_two_parts(self, file_path: str) -> str:
        parts = file_path.split(os.sep)
        if len(parts) < 2:
            raise ValueError("The file path must have at least two parts.")
        merged_name = f"{parts[-2]}_{parts[-1]}"
        return merged_name[:-4]

    def calculate_fft(self, data: np.ndarray, idx: int, file_name: str) -> np.ndarray:
        """
        due to symmetry in the FFT of real-valued signals, only the first N/2 points are unique and
        contain all the information needed to fully reconstruct the original signal.
        The second half of the FFT output is just the complex conjugate of the first half.
        """
        fft_result = fft(data)
        magnitude_spectrum = np.abs(fft_result[:len(fft_result) // 2])
        if idx == 1000:
            self.plot_fft(magnitude_spectrum, self.merge_last_two_parts(file_name))
        return magnitude_spectrum

    def plot_fft(self, fft_result: np.ndarray, file_name: str):
        plt.figure()
        plt.plot(fft_result)
        plt.title('FFT Amplitude Spectrum')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.savefig(os.path.join(self.save_plot, file_name + '.png'))
        plt.close()

    def process_files(self, filepath_df):
        csv_files = glob.glob(os.path.join(self.input_dir, 'acc_*.csv'))
        csv_files = sorted(csv_files)  # sorting to traverse them forward time order
        for idx, csv_file in enumerate(tqdm(csv_files, desc=f"Processing {str(self.input_dir).split('/')[-1]}")):
            try:
                with open(csv_file, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                    df = pd.read_csv(f, header=None, delimiter=dialect.delimiter)

                if df.shape[1] < 5:
                    raise ValueError(f"Skipping file {csv_file} due to insufficient columns")

                # df = self.correct_header(df)
                fft_result = np.array([self.calculate_fft(df.iloc[:, 4].values, idx, csv_file),
                              self.calculate_fft(df.iloc[:, 5].values, idx, csv_file)])
                base_name = os.path.basename(csv_file)
                npy_file = os.path.join(self.save_dir, base_name.replace('.csv', '.npy'))
                np.save(npy_file, fft_result)
                filepath_df.loc[len(filepath_df)] = [npy_file]

            except Exception as e:
                error_message = f"Error processing file {csv_file}: {str(e)}"
                logging.error(error_message)
                self.log_error(error_message)

    def correct_header(self, df: pd.DataFrame) -> pd.DataFrame:
        default_header = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']
        df.columns = default_header
        return df


if __name__ == "__main__":
    datasets = ['data/ieee-RUL/Full_Test_Set', 'data/ieee-RUL/Learning_set', 'data/ieee-RUL/Test_set']
    
    for dataset in datasets:
        filepath_df = pd.DataFrame(columns=['full_fft_path'])  # Create a new DataFrame for each dataset

        for data in glob.glob(os.path.join(dataset, 'Bearing*/')):
            processor = CSVProcessor(input_dir=data, save_dir=data[:-1] + "_fft/", save_plot='pics')
            processor.process_files(filepath_df)  # Ensure process_files modifies filepath_df in place

        # Save CSV for each dataset
        save_path = f"{dataset}_fft_filepaths.csv"
        filepath_df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
