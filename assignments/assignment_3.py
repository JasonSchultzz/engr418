import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
To detect the presence or absence of a specific substances in lab samples, 
an engineer intends to develop a spectrometer. The spectrometer shines a light 
through the sample, and measures the intensity of light at different frequencies 
after it propagates through the sample. To test this concept, the engineer 
collects data from different samples, some including the substance of interest 
(class 1), and some not including it (class 0). The optical intensities are 
collected in a dataset consisting of 8 features (8 frequencies) and 20 samples. 
The data set is given in the following file.

Train a classifier that learns to separate the two classes using all 8 features.
In your submission include the code, in addition to the confusion matrix and the learned weights. 
'''