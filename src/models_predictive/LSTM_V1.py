'''
    This is the test model 
'''

import pandas as pd
import tensorflow as tf
import keras
import subprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    # Source - https://stackoverflow.com/a/67504607
    # Posted by Briar Campbell, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-05-27, License - CC BY-SA 4; Check for Nvidia GPU
    try:
        subprocess.check_output('nvidia-smi')
        print(pd.DataFrame(subprocess.check_output('nvidia-smi').decode('utf-8').split('\n')))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('Error GPU detection: No Nvidia GPU in system.')
    except Exception as e:
        print(f"Error GPU detection: {e}")