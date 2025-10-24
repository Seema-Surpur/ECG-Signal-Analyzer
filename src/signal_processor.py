from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class SignalProcessor:
    """Class for processing and analyzing extracted ECG signals."""
    
    def __init__(self):
        self.PAPER_SPEED = 25  # mm/s (standard ECG paper speed)
        self.GRID_SIZE = 1  # mm (small grid size)
        self.VOLTAGE_PER_GRID = 0.1  # mV per mm (standard calibration)
        self.PIXELS_PER_MM = 10  # Estimated pixels per millimeter in the image
    
    def normalize_signal(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the signal coordinates and convert to physical units.
        
        Args:
            x (np.ndarray): X coordinates in pixels
            y (np.ndarray): Y coordinates in pixels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time (seconds) and voltage (mV) arrays
        """
        # Ensure inputs are numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Empty signal data")
            
        if len(x) != len(y):
            raise ValueError(f"Mismatched array lengths: x={len(x)}, y={len(y)}")
        
        # Sort points by x coordinate
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Remove duplicate x values by averaging corresponding y values
        unique_x, indices = np.unique(x, return_inverse=True)
        y_mean = np.zeros_like(unique_x, dtype=np.float64)
        np.add.at(y_mean, indices, y)
        counts = np.bincount(indices)
        y_mean = np.divide(y_mean, counts)  # Explicit divide to avoid integer division
        x, y = unique_x, y_mean
        
        try:
            # Import scipy at the beginning of the try block
            from scipy import signal
            
            # Convert x to time in seconds (25 mm/sec standard)
            x_min = np.min(x)
            x = (x - x_min) / (self.PAPER_SPEED * self.PIXELS_PER_MM)
            
            # Normalize y to standard ECG scaling (1 mV = 10 mm)
            y_max = np.max(y)
            y = y_max - y  # Invert the signal
            y = y - np.mean(y)  # Center around zero
            y = y / (self.PIXELS_PER_MM * 10)  # Scale to mV
            
            # Resample to uniform 500 Hz sampling rate
            target_fs = 500  # Hz
            t_new = np.arange(x[0], x[-1], 1/target_fs)
            if len(t_new) == 0:
                raise ValueError("No valid time points after resampling")
                
            y = np.interp(t_new, x, y)
            x = t_new.copy()  # Make a copy to ensure we have our own data
            
            # Apply bandpass filter (0.5 - 45 Hz)
            nyquist = target_fs / 2
            b, a = signal.butter(4, [0.5/nyquist, 45/nyquist], btype='band')
            y = signal.filtfilt(b, a, y)  # Apply filter forward and backward
            
            # Apply notch filter for 50/60 Hz interference
            b, a = signal.iirnotch(50, 30, target_fs)
            y = signal.filtfilt(b, a, y)
            
            # Adaptive baseline correction
            win_size = int(0.2 * target_fs)  # 200ms window
            if win_size < 1:
                win_size = 1
            if win_size % 2 == 0:
                win_size += 1
            baseline = signal.medfilt(y, win_size)
            y = y - baseline
            
        except Exception as e:
            raise ValueError(f"Error processing signal: {str(e)}")
        
        return x, y
    
    def resample_signal(self, x: np.ndarray, y: np.ndarray, 
                       sampling_rate: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the signal to a uniform sampling rate.
        
        Args:
            x (np.ndarray): Time points
            y (np.ndarray): Signal values
            sampling_rate (float): Desired sampling rate in Hz
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Resampled time points and signal values
        """
        # Calculate new time points
        t_start = x[0]
        t_end = x[-1]
        t_new = np.arange(t_start, t_end, 1.0/sampling_rate)
        
        # Interpolate signal values at new time points
        y_new = np.interp(t_new, x, y)
        
        return t_new, y_new
    
    def filter_signal(self, y: np.ndarray, sampling_rate: float = 500.0) -> np.ndarray:
        """
        Apply filters to remove noise from the signal.
        
        Args:
            y (np.ndarray): Signal values
            sampling_rate (float): Sampling rate in Hz
            
        Returns:
            np.ndarray: Filtered signal values
        """
        # TODO: Implement proper filtering (bandpass, notch filters etc.)
        # For now, just apply simple moving average
        window_size = int(sampling_rate * 0.01)  # 10ms window
        if window_size % 2 == 0:
            window_size += 1
        
        y_filtered = np.convolve(y, np.ones(window_size)/window_size, mode='same')
        
        return y_filtered
    
    def plot_signal(self, x: np.ndarray, y: np.ndarray, 
                   title: str = "ECG Signal", 
                   save_path: Optional[str] = None) -> None:
        """
        Plot the ECG signal with proper units and grid.
        
        Args:
            x (np.ndarray): Time points (seconds)
            y (np.ndarray): Signal values (mV)
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Create ECG paper style background
        plt.grid(True, which='major', color='r', linewidth=0.8, alpha=0.3)  # 5mm grid
        plt.grid(True, which='minor', color='r', linewidth=0.2, alpha=0.2)  # 1mm grid
        
        # Plot the signal
        plt.plot(x, y, 'b-', linewidth=1.2)
        
        # Set labels and title
        plt.title(title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Voltage (mV)")
        
        # Set standard ECG scaling
        plt.ylim(-1.0, 1.5)  # Show -1 to +1.5 mV range
        duration = min(3.0, x[-1] - x[0])  # Show up to 3 seconds
        plt.xlim(x[0], x[0] + duration)
        
        # Add minor ticks (1mm = 0.04s x 0.1mV)
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        
        # Add reference lines
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)  # Zero line
        plt.axhline(y=1, color='k', linestyle=':', alpha=0.2)  # 1mV reference
        plt.axhline(y=-0.5, color='k', linestyle=':', alpha=0.2)  # -0.5mV reference
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()