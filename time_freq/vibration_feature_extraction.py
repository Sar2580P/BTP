import numpy as np
from scipy.fft import fft
import pandas as pd
import os
import glob
import csv
import logging
from tqdm import tqdm
import pickle


class VibrationSignalFeatureExtractor:
    """
    A class for extracting time-domain and frequency-domain features from signals.
    """
    
    def __init__(self, input_dir: str, save_dir: str):
        """
        Initialize the feature extractor with an optional signal.
        
        Parameters:
        -----------
        signal : array-like, optional
            The input signal for feature extraction
        """
        self.input_dir = input_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
            
    def set_signal(self, signal):
        """
        Set the signal for feature extraction.
        
        Parameters:
        -----------
        signal : array-like
            The input signal for feature extraction
        """
        self.signal = signal
    
    # Time-domain features
    
    def mean(self, signal=None):
        """Calculate the mean value of the signal."""
        x = self.signal if signal is None else signal
        return np.mean(x)
    
    def rms(self, signal=None):
        """Calculate the Root Mean Square (RMS) value of the signal."""
        x = self.signal if signal is None else signal
        return np.sqrt(np.mean(np.square(x)))
    
    def variance(self, signal=None):
        """Calculate the variance of the signal."""
        x = self.signal if signal is None else signal
        n = len(x)
        mean_val = self.mean(x)
        return np.sqrt(np.sum((x - mean_val) ** 2) / (n - 1))
    
    def square_root_amplitude(self, signal=None):
        """Calculate the Square Root Amplitude value of the signal."""
        x = self.signal if signal is None else signal
        return (np.sum(np.abs(x)) / len(x)) ** 2
    
    def absolute_mean_amplitude(self, signal=None):
        """Calculate the Absolute Mean Amplitude value of the signal."""
        x = self.signal if signal is None else signal
        return np.mean(np.abs(x))
    
    def peak(self, signal=None):
        """Calculate the peak value of the signal."""
        x = self.signal if signal is None else signal
        return np.max(np.abs(x))
    
    def kurtosis(self, signal=None):
        """Calculate the kurtosis of the signal."""
        x = self.signal if signal is None else signal
        n = len(x)
        mean_val = self.mean(x)
        numerator = n * np.sum((x - mean_val) ** 4)
        denominator = (np.sum((x - mean_val) ** 2)) ** 2
        return numerator / denominator
    
    def maximum(self, signal=None):
        """Calculate the maximum value of the signal."""
        x = self.signal if signal is None else signal
        return np.max(x)
    
    def minimum(self, signal=None):
        """Calculate the minimum value of the signal."""
        x = self.signal if signal is None else signal
        return np.min(x)
    
    def peak_to_peak(self, signal=None):
        """Calculate the Peak-to-Peak value of the signal."""
        x = self.signal if signal is None else signal
        return self.maximum(x) - self.minimum(x)
    
    def skewness(self, signal=None):
        """Calculate the skewness of the signal."""
        x = self.signal if signal is None else signal
        n = len(x)
        mean_val = self.mean(x)
        numerator = np.sum((x - mean_val) ** 3) / n
        denominator = (np.sum((x - mean_val) ** 2) / n) ** 1.5
        return numerator / denominator
    
    def shape_factor(self, signal=None):
        """Calculate the Shape Factor of the signal."""
        x = self.signal if signal is None else signal
        return self.rms(x) / self.absolute_mean_amplitude(x)
    
    def crest_factor(self, signal=None):
        """Calculate the Crest Factor of the signal."""
        x = self.signal if signal is None else signal
        return self.peak(x) / self.rms(x)
    
    def impulse_factor(self, signal=None):
        """Calculate the Impulse Factor of the signal."""
        x = self.signal if signal is None else signal
        return self.peak(x) / self.absolute_mean_amplitude(x)
    
    def clearance_factor(self, signal=None):
        """Calculate the Clearance Factor of the signal."""
        x = self.signal if signal is None else signal
        return self.peak(x) / self.square_root_amplitude(x)
    
    # Frequency-domain features
    
    def frequency_center(self, signal=None, freq=None):
        """
        Calculate the Frequency Center of the signal.
        
        Parameters:
        -----------
        signal : array-like, optional
            The input signal in frequency domain
        freq : array-like, optional
            The frequency values corresponding to the signal
            
        Returns:
        --------
        float
            The frequency center
        """
        x = self.signal if signal is None else signal
        
        if freq is None:
            n = len(x)
            freq = np.fft.fftfreq(n) * n  # Generate frequency values if not provided
        
        numerator = np.sum(freq * x ** 2)
        denominator = 2 * np.pi * np.sum(x ** 2)
        
        return numerator / denominator if denominator != 0 else 0
    
    def rms_variance_frequency(self, signal=None, freq=None):
        """
        Calculate the RMS Variance Frequency of the signal.
        
        Parameters:
        -----------
        signal : array-like, optional
            The input signal in frequency domain
        freq : array-like, optional
            The frequency values corresponding to the signal
            
        Returns:
        --------
        float
            The RMS variance frequency
        """
        x = self.signal if signal is None else signal
        
        if freq is None:
            n = len(x)
            freq = np.fft.fftfreq(n) * n  # Generate frequency values if not provided
        
        numerator = np.sum(freq ** 2 * x ** 2)
        denominator = 4 * np.pi ** 2 * np.sum(x ** 2)
        
        return np.sqrt(numerator / denominator) if denominator != 0 else 0
    
    def root_variance_frequency(self, signal=None, freq=None):
        """
        Calculate the Root Variance Frequency of the signal.
        
        Parameters:
        -----------
        signal : array-like, optional
            The input signal in frequency domain
        freq : array-like, optional
            The frequency values corresponding to the signal
            
        Returns:
        --------
        float
            The root variance frequency
        """
        x = self.signal if signal is None else signal
        
        rms_var_freq = self.rms_variance_frequency(x, freq)
        freq_center = self.frequency_center(x, freq)
        
        return np.sqrt(rms_var_freq ** 2 - freq_center ** 2)
    
    def extract_all_time_features(self, signal=None):
        """
        Extract all time domain features from the signal.
        
        Parameters:
        -----------
        signal : array-like, optional
            The input signal
            
        Returns:
        --------
        dict
            Dictionary containing all time domain features
        """
        x = self.signal if signal is None else signal
        
        features = np.array([
            self.mean(x),
            self.rms(x),
            self.variance(x),
            self.square_root_amplitude(x),
            self.absolute_mean_amplitude(x),
            self.peak(x),
            self.kurtosis(x),
            self.maximum(x),
            self.minimum(x),
            self.peak_to_peak(x),
            self.skewness(x),
            self.shape_factor(x),
            self.crest_factor(x),
            self.impulse_factor(x),
            self.clearance_factor(x)
        ])
        
        return features
    
    def extract_all_frequency_features(self, signal=None, freq=None):
        """
        Extract all frequency domain features from the signal.
        
        Parameters:
        -----------
        signal : array-like, optional
            The input signal in frequency domain
        freq : array-like, optional
            The frequency values corresponding to the signal
            
        Returns:
        --------
        dict
            Dictionary containing all frequency domain features
        """
        x = self.signal if signal is None else signal
        
        features = np.array([
            self.frequency_center(x, freq),
            self.rms_variance_frequency(x, freq),
           self.root_variance_frequency(x, freq)
        ])
        
        return features
    
    def calculate_fft(self, data: np.ndarray) -> np.ndarray:
        """
        due to symmetry in the FFT of real-valued signals, only the first N/2 points are unique and 
        contain all the information needed to fully reconstruct the original signal. 
        The second half of the FFT output is just the complex conjugate of the first half.
        """
        fft_result = fft(data)
        magnitude_spectrum = np.abs(fft_result[:len(fft_result) // 2])
        return magnitude_spectrum
    

    def process_files(self):
        csv_files = glob.glob(os.path.join(self.input_dir, 'acc_*.csv'))
        csv_files = sorted(csv_files)  # sorting to traverse them forward time order
        start_time = None
        for idx, csv_file in enumerate(tqdm(csv_files, desc=f"Processing {str(self.input_dir).split('/')[-1]}")):
            try:
                with open(csv_file, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                    df = pd.read_csv(f, header=None, delimiter=dialect.delimiter)
                
                curr_time = df.iloc[0, 0]*3600 + df.iloc[0, 1]*60 + df.iloc[0, 2]   # time in seconds
                if start_time is None: start_time = curr_time
                
                acc_h , acc_v = df.iloc[:, 4].values, df.iloc[:, 5].values
                acc_h_fft, acc_v_fft = self.calculate_fft(acc_h), self.calculate_fft(acc_v)
                acc_h_time_features, acc_v_time_features = self.extract_all_time_features(acc_h), self.extract_all_time_features(acc_v)
                acc_h_freq_features, acc_v_freq_features = self.extract_all_frequency_features(acc_h_fft), self.extract_all_frequency_features(acc_v_fft)
                
                data_dict = {}
                data_dict['time'] = curr_time - start_time
                data_dict['acc_h'] = acc_h
                data_dict['acc_v'] = acc_v
                data_dict['acc_h_fft'] = acc_h_fft
                data_dict['acc_v_fft'] = acc_v_fft
                data_dict['acc_h_time_features'] = acc_h_time_features
                data_dict['acc_v_time_features'] = acc_v_time_features
                data_dict['acc_h_freq_features'] = acc_h_freq_features
                data_dict['acc_v_freq_features'] = acc_v_freq_features
                
                # save to pickle
                save_file = os.path.join(self.save_dir, f"vibration_features_{idx}.pkl")
                with open(save_file, 'wb') as f:
                    pickle.dump(data_dict, f)
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
        for data in glob.glob(os.path.join(dataset, 'Bearing*[0-9]/')):
            processor = VibrationSignalFeatureExtractor(input_dir=data, save_dir=data[:-1] + "_vibration_FE")
            processor.process_files()