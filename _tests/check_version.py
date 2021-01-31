# import pandas as pd
# import tensorflow as tf
# print(f'tensorflow version: {tf.__version__}')

# from tensorflow import keras
# print(f'keras version: {keras.__version__}')


from tensorflow.python.client import device_lib
print(f'Device: {device_lib.list_local_devices()}')