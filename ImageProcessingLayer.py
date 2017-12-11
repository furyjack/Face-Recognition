import cv2
import numpy as np
import face_recognition
from DatabaseLayer import Database_Layer
import matplotlib.pyplot as plt

class Image_Processing():
    face_encodings=[]
    training_labels=[]
    database_conn=Database_Layer()

    def __init__(self,training_data,training_labels):
        images=[face_recognition.load_image_file(x) for x in training_data]
        face_locations=[]
        fault=False
        for i,x in enumerate(images):
            try:
                face_locations.append(face_recognition.face_locations(x)[0:1])
            except:
                print('face not found in image')
                plt.imshow(x)
                plt.show()
                fault=True
                pass
        if fault:
             return
        self.face_encodings=[face_recognition.face_encodings(x,y) for x,y in zip(images,face_locations)]
        self.training_labels=training_labels
        self.conv_to_np()
        self.database_conn.save(self.face_encodings,training_labels)



    def conv_to_np(self):
        self.training_labels=np.asarray(self.training_labels).reshape(1,-1)
        self.face_encodings=[np.reshape(np.asarray(x),(1,128)) for x in self.face_encodings]
        self.face_encodings=np.concatenate(self.face_encodings)








