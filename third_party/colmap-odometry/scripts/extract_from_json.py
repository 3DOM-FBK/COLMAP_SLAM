import json
import os
import sys
import re

def extract_accelerometer_data(json_file_path, output_file_path):
    """
    Extract acc_x, acc_y, acc_z values from trace.json and write each value on a separate line.
    """
    
    all_values = []
    
    try:
        # Read the JSON file - it's a non-standard format with comma-separated JSON objects
        with open(json_file_path, 'r') as file:
            content = file.read()
        
        # Try to parse as standard JSON array first
        try:
            if content.startswith('[') and content.endswith(']'):
                data = json.loads(content)
            else:
                # Handle comma-separated JSON objects format
                # Remove leading comma if present and wrap in array brackets
                if content.startswith(','):
                    content = content[1:]
                if not content.startswith('['):
                    content = '[' + content + ']'
                data = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to parse line by line
            lines = content.strip().split('\n')
            data = []
            for line in lines:
                line = line.strip()
                if line.startswith(','):
                    line = line[1:]
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        print(f"Successfully loaded {json_file_path} with {len(data)} entries")
        
        # Function to recursively search for accelerometer data
        def find_accelerometer_values(obj):
            if isinstance(obj, dict):
                # Check if this object has a value that contains AiPntResult
                if 'value' in obj and isinstance(obj['value'], str) and 'AiPntResult' in obj['value']:
                    # Parse the AiPntResult string for accelerometer data
                    result_string = obj['value']
                    
                    # Use regex to extract acc_x, acc_y, acc_z values
                    acc_x_match = re.search(r'acc_x=np\.float64\(([-\d.e]+)\)', result_string)
                    acc_y_match = re.search(r'acc_y=np\.float64\(([-\d.e]+)\)', result_string)
                    acc_z_match = re.search(r'acc_z=np\.float64\(([-\d.e]+)\)', result_string)
                    
                    if acc_x_match and acc_y_match and acc_z_match:
                        acc_x = float(acc_x_match.group(1))
                        acc_y = float(acc_y_match.group(1))
                        acc_z = float(acc_z_match.group(1))
                        
                        all_values.append([acc_x, acc_y, acc_z])
                        print(f"Found accelerometer data: x={acc_x}, y={acc_y}, z={acc_z}")
                
                # Check if this object has all three accelerometer values as separate keys
                if 'acc_x' in obj and 'acc_y' in obj and 'acc_z' in obj:
                    all_values.append([obj['acc_x'], obj['acc_y'], obj['acc_z']])
                    print(f"Found direct accelerometer data: x={obj['acc_x']}, y={obj['acc_y']}, z={obj['acc_z']}")
                
                # Continue searching in nested objects
                for value in obj.values():
                    find_accelerometer_values(value)
                    
            elif isinstance(obj, list):
                # Search in list items
                for item in obj:
                    find_accelerometer_values(item)
        
        # Start the recursive search
        find_accelerometer_values(data)
        
        # Write to file in CSV format with headers
        if all_values:
            with open(output_file_path, 'w') as file:
                # Write CSV header
                file.write("acc_x,acc_y,acc_z\n")
                
                # Write data rows
                for acc_x, acc_y, acc_z in all_values:
                    file.write(f"{acc_x},{acc_y},{acc_z}\n")
            
            print(f"Successfully extracted {len(all_values)} accelerometer readings")
            print(f"Data written to: {output_file_path}")
            
        else:
            print("No accelerometer data found in the JSON file")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    # Define file paths
    json_file_path = '/home/threedom/Desktop/github_3dom/COLMAP_SLAM/_DATA/DIFFICULT_SCENE/output/trace.json'
    output_file_path = '/home/threedom/Desktop/github_3dom/COLMAP_SLAM/_DATA/DIFFICULT_SCENE/output/accelerometer_data.txt'
    
    # Check if trace.json exists
    if not os.path.exists(json_file_path):
        print(f"Error: trace.json not found at {json_file_path}")
        sys.exit(1)
    
    print("Extracting accelerometer data from trace.json...")
    extract_accelerometer_data(json_file_path, output_file_path)

if __name__ == "__main__":
    main()