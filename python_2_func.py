import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

print("b.1")
THRESHOLD_HEAD_POSE_COUNTER = 20
ODD_EVEN_COUNTER = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
print("b.2")
def python_2_func(shared_dict):
    print("b.3")
    COUNTER = 0
    ALARM_ON = False
    while shared_dict['run']:
        print("b.4")
        frame2_original = shared_dict['frame2']
        if frame2_original is not None:
            frame2 = frame2_original.copy()
            print("b.5")
            start = time.time()
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame2.flags.writeable = False
            results = face_mesh.process(frame2)
            frame2.flags.writeable = True
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = frame2.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                print("b.6")
                
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    print("b.7")

                    focal_length = 1 * img_w
                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])
                    
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    print("b.8")

                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360
                    print("b.9")

                    if y < -18 or y > 18 or x < -18 or x > 18 :
                        print("b.10")
                        COUNTER += 1
                        if (COUNTER > THRESHOLD_HEAD_POSE_COUNTER) :
                            print("b.11")
                            
                            if not ALARM_ON:
                                print("b.12")
                                ALARM_ON = True
                                shared_dict['alarm'] = 'alarm.wav'
                                shared_dict['ALARM_ON'] = ALARM_ON
                            if shared_dict['ALARM_ON']:
                                print("b.13")
                                pygame.mixer.init()
                                pygame.mixer.music.load(shared_dict['alarm'])
                                pygame.mixer.music.play(2)
                                shared_dict['ALARM_ON'] = False
                                # draw an alarm on the frame
                                cv2.putText(frame2, "LOOK STRAIGHT ! BE ALERT !", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    else:
                        ## text = "Forward"
                        print("b.14")
                        COUNTER = 0
                        ALARM_ON = False
                        

                    
                    print("b.15")

                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
                    print("b.16")
                    

                    cv2.line(frame2, p1, p2, (255, 0, 0), 3)


                    ##cv2.putText(frame2, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                
                cv2.putText(frame2, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                print("b.17")

           
            
            
            shared_dict['frame2'] = frame2
                
            print("b.18")
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
        else:
            print("b.19")
