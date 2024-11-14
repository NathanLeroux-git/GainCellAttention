import json
import os

def json_to_py(json_file, py_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    with open(py_file, 'w') as f:
        # f.write("# Auto-generated Python file from JSON\n")
        f.write("experiment_config = {\n")
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            else:
                f.write(f"    '{key}': {value},\n")
        f.write("}\n")
        # f.write("\nexperiment_config['Net'] = experiment_config['trainingNet']")
        # f.write("\nexperiment_config['test_attention'] = experiment_config['train_attention']")

# Path to directory containing JSON files
json_dir = 'configs/experiments/online_transformers_biocas_2023/'

# Path to directory where Python files will be saved
py_dir = json_dir

# Ensure the directory to save Python files exists
if not os.path.exists(py_dir):
    os.makedirs(py_dir)

# Convert each JSON file to a Python file
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        json_file = os.path.join(json_dir, filename)
        py_file = os.path.join(py_dir, os.path.splitext(filename)[0] + '.py')
        json_to_py(json_file, py_file)