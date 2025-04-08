import pickle
import numpy as np
import os
import argparse

def convert_precision(input_file, output_file=None, precision='float32'):
    """
    Convert NumPy arrays in a pickle file to a specified precision.
    
    Args:
        input_file (str): Path to the input pickle file
        output_file (str, optional): Path to save the output file. If None, will modify the original filename.
        precision (str): Target precision - 'float32' or 'float64'
    """
    if precision not in ['float32', 'float64']:
        raise ValueError("Precision must be either 'float32' or 'float64'")
    
    # Create output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_{precision}{ext}"
    
    print(f"Loading data from {input_file}")
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False
    
    # Check if data is a numpy array
    if isinstance(data, np.ndarray):
        if data.dtype in [np.float32, np.float64]:
            current_dtype = str(data.dtype)
            print(f"Converting array from {current_dtype} to {precision}")
            print(f"Original shape: {data.shape}")
            
            # Convert precision
            if precision == 'float32':
                data = data.astype(np.float32)
            else:
                data = data.astype(np.float64)
            
            print(f"New dtype: {data.dtype}")
        else:
            print(f"Warning: Array is not float type (current type: {data.dtype})")
            if input("Convert anyway? (y/n): ").lower() == 'y':
                # Convert precision
                if precision == 'float32':
                    data = data.astype(np.float32)
                else:
                    data = data.astype(np.float64)
                print(f"Converted to {data.dtype}")
            else:
                print("Conversion cancelled")
                return False
    else:
        print(f"Warning: Data is not a NumPy array (type: {type(data)})")
        return False
    
    # Save converted data
    print(f"Saving converted data to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print("Conversion completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert precision of NumPy arrays in pickle files')
    parser.add_argument('input_file', help='Input pickle file path')
    parser.add_argument('-o', '--output_file', help='Output file path (optional)', default=None)
    parser.add_argument('-p', '--precision', choices=['float32', 'float64'], 
                        default='float32', help='Target precision (default: float32)')
    
    args = parser.parse_args()
    
    print("===== Pickle Precision Converter =====")
    convert_precision(args.input_file, args.output_file, args.precision)

if __name__ == "__main__":
    main()
