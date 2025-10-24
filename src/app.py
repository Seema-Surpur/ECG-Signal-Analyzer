from flask import Flask, request, render_template, jsonify, make_response
from werkzeug.utils import secure_filename
from image_processor import ECGImageProcessor
from signal_processor import SignalProcessor
import os
import numpy as np
import base64
import json
import csv
from io import BytesIO, StringIO
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processors
img_processor = ECGImageProcessor()
sig_processor = SignalProcessor()

def process_ecg_image(image_path):
    """
    Process an ECG image and return the processed signal data
    """
    try:
        # Extract raw signal from image
        print("Extracting signal from image...")
        x, y = img_processor.process_image(image_path)
        if x is None or y is None:
            raise ValueError("Failed to extract signal from image")
            
        # Convert to numpy arrays if they aren't already
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        print(f"Raw signal extracted: {len(x)} points")
        
        # Process the signal
        print("Normalizing signal...")
        x, y = sig_processor.normalize_signal(x, y)
        
        print("Resampling signal...")
        x, y = sig_processor.resample_signal(x, y)
        
        print("Filtering signal...")
        y = sig_processor.filter_signal(y)
        
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            raise ValueError("Signal processing failed")
            
        print(f"Signal processing complete: {len(x)} points")
        return x, y
        
    except Exception as e:
        print(f"Error in process_ecg_image: {str(e)}")
        return None, None

def plot_to_base64(x, y):
    """Convert matplotlib plot to base64 string"""
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.title("ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    
    # Save plot to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    print("Received /process request")
    
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    # Print debugging info
    print(f"Processing file: {file.filename}")
    print(f"Content type: {file.content_type}")
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved temporarily as: {filepath}")

        # Initialize variables
        x = None
        y = None
        plot_data = None
        
        try:
            # Process the image and get signal data
            print("Processing image...")
            result_x, result_y = process_ecg_image(filepath)
            
            if result_x is None or result_y is None:
                raise ValueError("Signal processing failed - no data returned")
            
            # From this point on, we know we have valid data
            # Store processed data in local variables
            x_list = result_x.tolist() if hasattr(result_x, 'tolist') else list(result_x)
            y_list = result_y.tolist() if hasattr(result_y, 'tolist') else list(result_y)
            
            if not x_list or not y_list:
                raise ValueError("Empty signal data")
                
            if len(x_list) != len(y_list):
                raise ValueError(f"Mismatched data lengths: x={len(x_list)}, y={len(y_list)}")
            
            print(f"Signal extracted - Points: {len(x_list)}")
            print(f"Signal range - X: [{min(x_list):.2f}, {max(x_list):.2f}], Y: [{min(y_list):.2f}, {max(y_list):.2f}]")
            
            # Store data for downloads first
            print("Storing data in app config...")
            app.config['LAST_PROCESSED_DATA'] = {
                'x': x_list,
                'y': y_list
            }
            
            # Generate plot using the validated data
            print("Generating plot...")
            plot_data = plot_to_base64(result_x, result_y)
            
            # Create data points for display
            points = [
                {
                    'time': float(t),
                    'value': float(v),
                    'type': 'Original'
                } 
                for t, v in zip(x_list, y_list)
            ]
            
            print("Data processing complete")
            return jsonify({
                'success': True,
                'plot': plot_data,
                'message': 'ECG processed successfully',
                'points': points
            })
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error handling file: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temporary file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Temporary file removed: {filepath}")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/download/<format>')
def download_data(format):
    """Handle data downloads in CSV or JSON format"""
    print(f"Download request received for format: {format}")
    
    if format not in ['csv', 'json']:
        print("Invalid format requested")
        return 'Invalid format', 400
    
    # Get the last processed data
    try:
        data = app.config.get('LAST_PROCESSED_DATA')
        print(f"Retrieved data from config: {data is not None}")
        
        if not data or not isinstance(data, dict):
            print("No valid data in app config")
            return 'No data available', 404
            
        if 'x' not in data or 'y' not in data:
            print("Incomplete data in app config")
            return 'Data format invalid', 400
            
        # Get data from config and convert to lists if needed
        time_points = list(data['x'])
        signal_values = list(data['y'])
        
        if len(time_points) == 0 or len(signal_values) == 0:
            print("Empty data arrays")
            return 'No signal data available', 404
            
        if len(time_points) != len(signal_values):
            print(f"Mismatched array lengths: x={len(time_points)}, y={len(signal_values)}")
            return 'Data integrity error', 500
            
        print(f"Data retrieved - points: {len(time_points)}")
        
        try:
            if format == 'csv':
                # Create CSV string
                output = StringIO()
                writer = csv.writer(output)
                writer.writerow(['Time', 'Value', 'Type'])
                
                # Write data points with proper formatting
                for t, v in zip(time_points, signal_values):
                    writer.writerow([f"{float(t):.3f}", f"{float(v):.3f}", 'Original'])
                
                response = make_response(output.getvalue())
                response.headers['Content-Type'] = 'text/csv'
                response.headers['Content-Disposition'] = 'attachment; filename=ecg_data.csv'
                
            else:  # JSON
                # Create JSON data with explicit float conversion
                json_data = {
                    'points': [
                        {
                            'time': float(t),
                            'value': float(v),
                            'type': 'Original'
                        }
                        for t, v in zip(time_points, signal_values)
                    ]
                }
                
                response = make_response(json.dumps(json_data, indent=2))
                response.headers['Content-Type'] = 'application/json'
                response.headers['Content-Disposition'] = 'attachment; filename=ecg_data.json'
            
            print(f"Download response prepared: {format}")
            return response
            
        except Exception as e:
            print(f"Error formatting download file: {str(e)}")
            return f'Error creating download file: {str(e)}', 500
            
    except Exception as e:
        print(f"Error accessing data: {str(e)}")
        return f'Error accessing data: {str(e)}', 500


if __name__ == '__main__':
    app.run(debug=True)