# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
from collections import deque
from threading import Thread
import numpy as np
import pygame
import time
import dlib
import cv2
import imutils

print("a.1")
DURATION = 4  # in seconds
ear_data = deque() 
start = time.time()
print("a.2")

def python_1_func(shared_dict):
    print("a.3")
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    p = "shape_predictor_68_face_landmarks.dat"
    time.sleep(0.01)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    print("a.4")

    # initialize the frame counters and the total number of blinks
    ALARM_ON = False
    EYE_AR_THRESH = 0.16
    

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def sound_alarm(path):
        print("a.5")
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(2)

        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick()

    def eye_aspect_ratio(eye):
        print("a.6")
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    while shared_dict['run']:
        print("a.7")
        frame1 = shared_dict['frame1']
        
        if frame1 is not None:
            print("a.8")
            
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            

            for rect in rects:
                print("a.9")
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0


                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                time.sleep(0.01)
                cv2.drawContours(frame1, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame1, [rightEyeHull], -1, (0, 255, 0), 1)

                ear_data.append((time.time(), ear))

                # Remove EAR values that are older than DURATION
                while ear_data and (time.time() - ear_data[0][0]) > DURATION:
                    ear_data.popleft()

                # Calculate the average EAR value for the past DURATION
                avg_ear = sum([data[1] for data in ear_data]) / len(ear_data)
                print("a.10")

                if (avg_ear < EYE_AR_THRESH) :
                    print("a.11")
                    if time.time()-start >= DURATION :
                        print("a.12")
        
                        if not ALARM_ON:
                            print("a.13")
                            ALARM_ON = True
                            shared_dict['alarm'] = 'alarm.wav'
                            shared_dict['ALARM_ON'] = ALARM_ON
                        if shared_dict['ALARM_ON']:
                            print("a.14")
                            sound_alarm(shared_dict['alarm'])
                            shared_dict['ALARM_ON'] = False
                        # draw an alarm on the frame
                        cv2.putText(frame1, "DROWSINESS HAS BEEN DETECTED! BE ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    print("a.15")
                    ALARM_ON = False
            shared_dict['frame1'] = frame1
            print("a.16")

        
        print("a.17")
