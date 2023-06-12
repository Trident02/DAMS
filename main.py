from multiprocessing import Process, Manager
import time
import imutils
import cv2
from python_1_func import python_1_func
from python_2_func import python_2_func
print("1")
def capture_frame(shared_dict):
    print("2")
    cap = cv2.VideoCapture(0)
    while shared_dict['run']:
        print("3")
        ret, frame = cap.read()
        if not ret:
            break
        shared_dict['frame1'] = frame.copy()
        shared_dict['frame2'] = frame.copy()
        print("4")
        time.sleep(0.1)
        
    cap.release()

print("5")
if __name__ == "__main__":
    with Manager() as manager:
        shared_dict = manager.dict()
        shared_dict['run'] = True
        shared_dict['frame1'] = None
        shared_dict['frame2'] = None
        shared_dict['alarm'] = ""
        shared_dict['ALARM_ON'] = False
        print("6")

        p1 = Process(target=capture_frame, args=(shared_dict,))
        p1.start()

        p2 = Process(target=python_1_func, args=(shared_dict,))
        p3 = Process(target=python_2_func, args=(shared_dict,))
        p2.start()
        p3.start()
        print("7")

        try:
            while True:
                print("8")
                frame1 = shared_dict['frame1']
                
                frame2 = shared_dict['frame2']
                

                if frame1 is not None:
                    print("9")
                    time.sleep(0.5)
                    frame1 = imutils.resize(frame1, width=600)
                    cv2.imshow('Camera Feed 1', frame1)
                if frame2 is not None:
                    print("10")
                    time.sleep(0.5)
                    frame2 = imutils.resize(frame2, width=600)
                    cv2.imshow('Camera Feed 2', frame2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Stopping...")
            shared_dict['run'] = False
        
        print("11")

        p1.join()
        p2.join()
        p3.join()
