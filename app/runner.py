import os
import subprocess

fileDir = os.path.dirname(os.path.realpath(__file__))
subprocess.run(["streamlit", "run", "webapp.py"], cwd = fileDir)    