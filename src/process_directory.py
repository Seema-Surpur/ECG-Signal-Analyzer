import os
from pathlib import Path
from image_processor import ECGImageProcessor
from signal_processor import SignalProcessor

def process_directory(input_dir, output_dir):
    """
    Process all ECG images in a directory and its subdirectories.
    
    Args:
        input_dir (str): Path to input directory containing ECG images
        output_dir (str): Path to output directory for processed data
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    img_processor = ECGImageProcessor()
    sig_processor = SignalProcessor()
    
    # Walk through all files in input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get full paths
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = output_path / rel_path
                
                # Create output subdirectories if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    print(f"Processing {input_file}...")
                    
                    # Process image
                    x, y = img_processor.process_image(str(input_file))
                    
                    # Process signal
                    x, y = sig_processor.normalize_signal(x, y)
                    x, y = sig_processor.resample_signal(x, y)
                    y = sig_processor.filter_signal(y)
                    
                    # Save results
                    plot_file = output_file.with_suffix('.png')
                    data_file = output_file.with_suffix('.npz')
                    
                    sig_processor.plot_signal(x, y, save_path=str(plot_file))
                    import numpy as np
                    np.savez(data_file, time=x, signal=y)
                    
                    print(f"Saved to {plot_file} and {data_file}")
                    
                except Exception as e:
                    print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process ECG images in a directory")
    parser.add_argument("input_dir", help="Directory containing ECG images")
    parser.add_argument("output_dir", help="Directory to save processed data")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)