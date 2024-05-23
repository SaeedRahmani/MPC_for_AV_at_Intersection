import streamlit as st
import os
import subprocess
import json
import sys
import importlib.util

# Define the path to the configuration file
config_path = 'config.json'

# Load the current configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Define a dictionary with the file paths
file_paths = {
    'Plausibility': '../planner/motion_primitive_search_plausibility.py',
    'Intersection': '../scenarios/mpc_intersection.py'  # Replace with the actual path
}

# Use Streamlit's sidebar to create a selectbox for the file
file = st.sidebar.selectbox('File', list(file_paths.keys()))

# Depending on the selected file, show different options
if file == 'Plausibility':
    config['test_no'] = st.sidebar.slider('Test No', min_value=1, max_value=2, value=config.get('test_no', 1))
elif file == 'Intersection':
    config['start_pos'] = st.sidebar.slider('Start Pos', min_value=1, max_value=3, value=config.get('start_pos', 1))
    config['turn_indicator'] = st.sidebar.slider('Turn Indicator', min_value=1, max_value=3, value=config.get('turn_indicator', 1))

# Write the new configuration to the file
with open(config_path, 'w') as f:
    json.dump(config, f)

# Get the path to the selected file
file_path = file_paths[file]

# Create a button to run the file
if st.button('Run file'):
    spec = importlib.util.spec_from_file_location("main", file_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    result = foo.main()

    # Create a placeholder
    placeholder = st.empty()

    try:
        iter(result)
    except TypeError:
        # If result is not iterable, display it directly
        placeholder.pyplot(result)
    else:
        # If result is iterable, iterate over it and display each figure
        for fig in result:
            placeholder.pyplot(fig)