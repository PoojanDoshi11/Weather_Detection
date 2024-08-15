from nbconvert import PythonExporter
import nbformat

# Load the notebook file
with open('Untitled.ipynb', 'r') as notebook_file:
    notebook_content = nbformat.read(notebook_file, as_version=4)

# Create a Python exporter
python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook_content)

# Save the Python code to a .py file
with open('model.py', 'w') as python_file:
    python_file.write(python_code)
