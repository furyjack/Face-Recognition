import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from DatabaseLayer import Database_Layer

class ML_layer():
    model=None
    dbconn=Database_Layer()
    def __init__(self,type='log'):
        if type=='log':
            self.model=LogisticRegression(C=1000)
        elif type=='svm':
            self.model=LinearSVC(C=1000)


    def train(self):
        train_data,train_labels=self.dbconn.load()
        self.model.fit(train_data,train_labels)
        self.dbconn.save_model(self.model)



