# Face-Detection-Model-using-Caffemodel
  
I have added codes for detecting faces from webcam, as well as video files, and improving the fps rate compared to the naive process. For improvement, I have 
used another separate thread for read() operation of cv2, so that it wont block other operations from happening in the main thread, and hence fps will increase.
For executing the code-
  image for - just run face_detection.py
  video file code - python face_detection_vstream.py -v video2.mp4
  webcam video file - python3 fps_improved.py -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel -d 1 -n 200
  
In the webcam model, if you want to keep the video detection on according to your wish, the just change the while condition to - 
  while True:
  
This will help your code run the detection as long as you want.
