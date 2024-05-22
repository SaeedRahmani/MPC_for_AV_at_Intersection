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

# Use Streamlit's sidebar to create an input for test_no
config['test_no'] = st.sidebar.slider('Test No', min_value=1, max_value=3, value=config.get('test_no', 1))

# Write the new configuration to the file
with open(config_path, 'w') as f:
    json.dump(config, f)

# Define the path to the file you want to run
file_path = '../planner/motion_primitive_search_plausibility.py'

# Create a button to run the file
if st.button('Run file'):
    spec = importlib.util.spec_from_file_location("main", file_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    fig = foo.main()
    st.pyplot(fig)