import cv2
import numpy as np
import os
from ImageProcessingLayer import Image_Processing
from DatabaseLayer import Database_Layer
from MLlayer import ML_layer
import face_recognition

class Application_Layer():
    num_labels=0
    train_data=[]
    train_labels=[]
    imgProc=None
    dbConn=Database_Layer()
    mlconn=ML_layer()


    def __init__(self):
        pass

    def input(self,path_to_train,batch_size):
        self.train_data,self.train_labels=self.load_images_from_directory(path_to_train,batch_size)
        self.imgProc=Image_Processing(self.train_data,self.train_labels)
        self.mlconn.train()



    def generate_labels(self,label,num):
        r=[]
        for i in range(num):
            r.append(label)
        return  r

    def load_images_from_directory(self,path,batch_size):
        dirs=os.listdir(path)
        self.num_labels=len(dirs)

        images=[]
        m_=10000
        for dir in dirs:
            m_=min(m_,len(os.listdir(path+'/'+ dir)))
            images.append(['training_images/'+dir+'/'+x for x in os.listdir(path+'/'+dir)])

        train_data=[]
        train_labels=[]
        for j in range(0,m_,batch_size):
            for i in range(self.num_labels):
               next_ims=min(batch_size,m_-j)
               train_data += (images[i][j:j+next_ims])
               train_labels += self.generate_labels(i,next_ims)

        for i in range(self.num_labels):
            train_data +=(images[i][m_:])
            train_labels += self.generate_labels(i,len(images[i])-m_)

        return train_data,train_labels


    def start_video_stream(self,cam_num=0):
        video_capture=cv2.VideoCapture(cam_num)
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        clf=self.dbConn.load_model()
        names=os.listdir('training_images')

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Only process every other frame of video to save time
            if process_this_frame:
                pass
                #Find all the faces and face encodings in the current frame of video
                #face_ladmarks=face_recognition.face_landmarks(small_frame)
                face_locations = face_recognition.face_locations(small_frame,model='cnn')
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                face_names = []
                face_prob = []
                # scores=[]
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    match = clf.predict(np.asarray(face_encoding.reshape((1, 128))))
                    probs = clf.predict_proba(np.asarray(face_encoding).reshape((1, 128)))
                    # score=face_recognition.face_distance([obama_face_encoding], face_encoding)
                    name = "Unknown"
                    prob = np.max(probs[0])
                    name=names[match[0]]

                    face_names.append(name)
                    face_prob.append(prob)


            process_this_frame = not process_this_frame

            i = 0
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):

                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                #         for x in face_ladmarks[i]:
                #             pts=face_ladmarks[i][x]
                #             for j in pts:
                #                 cv2.circle(frame,(j[0]*4-400,j[1]*4),1,(255,0,0),2)



                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, str(face_prob[i]), (left + 6, bottom + 40), font, 1.0, (255, 255, 255), 1)
                # cv2.putText(frame, str(scores[i]), (left - 10, bottom - 40), font, 1.0, (255, 255, 255), 1)
                i += 1

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()


obj=Application_Layer()
#obj.input('training_images',batch_size=2)
obj.start_video_stream()




