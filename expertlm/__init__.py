"""
HumanityHelper package
MoE language model implementation with domain-specific expertise
"""

# Import submodules to make them accessible when importing the package
from expertlm import models
from expertlm import utils

# Define what symbols to expose when using `from src import *`
__all__ = ['models', 'utils']

# Define package metadata
__version__ = '0.1.0'
__author__ = 'Jeffrey Rivero'
__email__ = 'jeff@check-ai.com'
