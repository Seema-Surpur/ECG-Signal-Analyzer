import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class ECGImageProcessor:
    """Class for processing ECG images and extracting signal data."""
    
    def __init__(self):
        pass
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an ECG image.
        
        Args:
            image_path (str): Path to the ECG image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Check minimum size
        if image.shape[0] < 100 or image.shape[1] < 100:
            raise ValueError("Image is too small. Minimum size is 100x100 pixels.")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if image is too large
        max_size = 1500
        if gray.shape[0] > max_size or gray.shape[1] > max_size:
            scale = max_size / max(gray.shape[0], gray.shape[1])
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        
        return gray
    
    def enhance_signal(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the ECG signal in the image.
        
        Args:
            image (np.ndarray): Input grayscale image
            
        Returns:
            np.ndarray: Enhanced image
        """
        try:
            # Make a copy to avoid modifying the original
            img = image.copy()
            
            # Normalize image
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Denoise
            img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            
            # Apply Gaussian blur
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Apply Otsu's thresholding
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Clean up the binary image
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Remove small objects and fill small holes
            min_size = 100
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
            
            # Keep only large enough components
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    binary[labels == i] = 0
                    
            return binary
            
        except Exception as e:
            print(f"Error enhancing signal: {str(e)}")
            return None

        return clean
    
    def extract_signal(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time series data from the processed ECG image.
        
        Args:
            image (np.ndarray): Preprocessed binary image
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y coordinates of the signal
        """
        try:
            # Ensure we have a binary image
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours with simplified approximation
            contours, hierarchy = cv2.findContours(
                binary, 
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                print("No contours found in the image")
                return None, None
            
            # Filter contours by size and shape
            min_area = image.shape[0] * image.shape[1] * 0.001
            min_length = image.shape[1] * 0.1  # Minimum 10% of image width
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                length = cv2.arcLength(contour, closed=False)
                if area > min_area and length > min_length:
                    valid_contours.append(contour)
            
            if not valid_contours:
                print("No valid signal contours found")
                return None, None
            
            # Get the most likely ECG signal contour (longest valid contour)
            signal_contour = max(valid_contours, key=lambda c: cv2.arcLength(c, False))
            
            # Extract coordinates
            coords = signal_contour.squeeze()
            
            # Validate coordinates shape
            if len(coords.shape) < 2:
                print("Invalid contour shape detected")
                return None, None
                
            # Extract x and y coordinates
            x = coords[:, 0].astype(np.float64)
            y = coords[:, 1].astype(np.float64)
            
            # Remove duplicate x coordinates by averaging y values
            x_unique, indices = np.unique(x, return_inverse=True)
            y_mean = np.zeros_like(x_unique, dtype=np.float64)
            np.add.at(y_mean, indices, y)
            counts = np.bincount(indices)
            y_mean = np.divide(y_mean, counts)
            
            # Sort by x coordinate
            sort_idx = np.argsort(x_unique)
            x = x_unique[sort_idx]
            y = y_mean[sort_idx]
            
            if len(x) < 10:  # Require at least 10 points for a valid signal
                print("Too few points in extracted signal")
                return None, None
                
            return x, y
            
        except Exception as e:
            print(f"Error extracting signal: {str(e)}")
            return None, None
        
        return x, y
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an ECG image and extract the signal data.
        
        Args:
            image_path (str): Path to the ECG image file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: x and y coordinates of the signal
        """
        try:
            print(f"Loading image from {image_path}")
            image = self.load_image(image_path)
            if image is None:
                return None, None
                
            print("Enhancing signal...")
            enhanced = self.enhance_signal(image)
            if enhanced is None:
                return None, None
            
            print("Extracting signal...")
            x, y = self.extract_signal(enhanced)
            if x is None or y is None:
                return None, None
                
            print(f"Signal extracted successfully: {len(x)} points")
            return x, y
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None