import numpy as np
import datetime
import random

def filter_digits(features, labels, digits):
    labels = labels.flatten()
    mask = np.isin(labels, digits)
    return features[mask], labels[mask]

#idk what im doing tbh

def generate_int():
    timestamp = datetime.datetime.now()
    lowerbound = 1
    upperbound = 2047

    random.seed(timestamp.strftime('%f'))
    return random.randrange(lowerbound, upperbound)