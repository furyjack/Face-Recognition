
import numpy as np
from sklearn.externals import joblib


class Database_Layer():
    def __init__(self):
        pass


    def save(self,training_data,training_labels):
        np.save('data/training_data',training_data)
        np.save('data/training_labels',training_labels)

    def load(self):
        return np.load('data/training_data.npy'),np.load('data/training_labels.npy')

    def save_model(self,model):
        joblib.dump(model,'models/linmodel.pkl')

    def load_model(self):
        return joblib.load('models/linmodel.pkl')
