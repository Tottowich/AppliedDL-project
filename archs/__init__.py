# This is the __init__.py file in the archs folder:

# Path: archs/__init__.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from .segmentation import *
