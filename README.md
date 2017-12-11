# Face-Recognition
<b>Dependencies</b>

scikit-learn

face-recognition

numpy

matplotlib

cv2

<b>Setup:</b>

Your directory where you unzip this code should look as follows:

Face-Recogniton

--training_images

----subject1   //name the folder on the name of the person whose images are contained in the folder

----subject2

--models

--data

--.py files


<b>Then run the following commands:</b>

obj=Application_Layer()

obj.input('training_images',batch_size=2)

obj.start_video_stream()
