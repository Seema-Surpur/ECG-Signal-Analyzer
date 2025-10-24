from pathlib import Path
from image_processor import ECGImageProcessor
from signal_processor import SignalProcessor
import matplotlib.pyplot as plt

def process_ecg_image(image_path: str, output_dir: str):
    """
    Process an ECG image and save the extracted digital signal.
    
    Args:
        image_path (str): Path to the input ECG image
        output_dir (str): Directory to save the output files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    img_processor = ECGImageProcessor()
    sig_processor = SignalProcessor()
    
    # Extract raw signal from image
    x, y = img_processor.process_image(image_path)
    
    # Process the signal
    x, y = sig_processor.normalize_signal(x, y)
    x, y = sig_processor.resample_signal(x, y)
    y = sig_processor.filter_signal(y)
    
    # Save the processed signal plot
    output_file = output_path / f"{Path(image_path).stem}_processed.png"
    sig_processor.plot_signal(x, y, save_path=str(output_file))
    
    # Save the signal data
    import numpy as np
    data_file = output_path / f"{Path(image_path).stem}_data.npz"
    np.savez(data_file, time=x, signal=y)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ECG images to digital signals")
    parser.add_argument("image_path", type=str, help="Path to the input ECG image")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    
    args = parser.parse_args()
    process_ecg_image(args.image_path, args.output_dir)