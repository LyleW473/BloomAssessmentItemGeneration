"""
Path setup helper for the assessment generation scripts.
- Appends the project root to `sys.path` so that `src.*` modules can be imported when a script is run directly.
- Imported for its side effect at the top of each script in this directory.
"""
import sys
import os

# Get the directory of this script (new_project/scripts)
script_dir = os.path.dirname(os.path.realpath(__file__))

# One level up (new_project/scripts -> new_project), which contains the `src` package
project_root = os.path.dirname(script_dir)

sys.path.append(project_root)
