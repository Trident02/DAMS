import cv2
import dlib
import numpy as np
from collections import deque
import time

# Load the shape predictor model
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Initialize the dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize variables for pupil detection
pupil_history = []  # List to store the history of pupil positions
num_frames_accumulated = 2  # Number of frames to accumulate for stabilization

# Initialize the deque to store pupil positions for the last 10 seconds
position_history = deque()  # Assuming 10 frames per second


# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks (x, y) coordinates
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Compute the Euclidean distance between the horizontal eye landmark (x, y) coordinates
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# Function to detect and locate pupils
def detect_pupils(image, shape):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the eye aspect ratio for both eyes
    left_eye_landmarks = np.array(shape[36:42])
    right_eye_landmarks = np.array(shape[42:48])
    left_ear = eye_aspect_ratio(left_eye_landmarks)
    right_ear = eye_aspect_ratio(right_eye_landmarks)

    # Define the thresholds for closed eyes
    ear_threshold = 0.2

    # Check if the eyes are closed or partially closed
    if left_ear < ear_threshold:
        left_eye_landmarks = np.concatenate([left_eye_landmarks, [shape[36]]])
    if right_ear < ear_threshold:
        right_eye_landmarks = np.concatenate([right_eye_landmarks, [shape[42]]])

    # Calculate the eye region of interest (ROI)
    left_eye_roi = cv2.boundingRect(left_eye_landmarks)
    right_eye_roi = cv2.boundingRect(right_eye_landmarks)

    # Get the eye regions from the grayscale image
    left_eye_region = gray[left_eye_roi[1]:left_eye_roi[1] + left_eye_roi[3],
                      left_eye_roi[0]:left_eye_roi[0] + left_eye_roi[2]]
    right_eye_region = gray[right_eye_roi[1]:right_eye_roi[1] + right_eye_roi[3],
                       right_eye_roi[0]:right_eye_roi[0] + right_eye_roi[2]]

    # Apply adaptive thresholding to enhance the pupil/iris contrast
    _, left_eye_thresh = cv2.threshold(left_eye_region, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, right_eye_thresh = cv2.threshold(right_eye_region, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the thresholded images
    left_eye_contours, _ = cv2.findContours(left_eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_eye_contours, _ = cv2.findContours(right_eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area in each eye region
    if len(left_eye_contours) > 0:
        left_eye_contour = max(left_eye_contours, key=cv2.contourArea)
        ((x, y), left_eye_radius) = cv2.minEnclosingCircle(left_eye_contour)
        left_eye_center = (int(x) + left_eye_roi[0], int(y) + left_eye_roi[1])
    else:
        left_eye_center = (left_eye_roi[0] + left_eye_roi[2] // 2, left_eye_roi[1] + left_eye_roi[3] // 2)

    if len(right_eye_contours) > 0:
        right_eye_contour = max(right_eye_contours, key=cv2.contourArea)
        ((x, y), right_eye_radius) = cv2.minEnclosingCircle(right_eye_contour)
        right_eye_center = (int(x) + right_eye_roi[0], int(y) + right_eye_roi[1])
    else:
        right_eye_center = (right_eye_roi[0] + right_eye_roi[2] // 2, right_eye_roi[1] + right_eye_roi[3] // 2)

    return left_eye_center, right_eye_center



# Function to classify the position of the pupil
def classify_pupil_position(avg_left_eye, avg_right_eye, displacement_threshold):
    left_displacement = avg_left_eye[0] - left_eye[0]
    right_displacement = right_eye[0] - avg_right_eye[0]

    left_eye_landmarks = np.array(shape[36:42])
    right_eye_landmarks = np.array(shape[42:48])


    # Check if the average pupil position is closer to contour points 40 and 46 (left eye)
    if ((np.linalg.norm(right_eye_landmarks[3]-avg_right_eye))-(np.linalg.norm(avg_right_eye - right_eye_landmarks[0]))) > displacement_threshold:
        return "Right"
    # Check if the average pupil position is closer to contour points 37 and 43 (right eye)
    elif ((np.linalg.norm(avg_right_eye - right_eye_landmarks[0])) - (np.linalg.norm(right_eye_landmarks[3]-avg_right_eye))) > displacement_threshold:
        return "Left"
    else:
        return "Center"




# Open a video capture object to capture video from a webcam (index 0)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video capture

    if ret:
        # Detect faces in the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = detector(gray)  # Use the face detector to detect faces in the grayscale frame

        for face in faces:
            # Determine facial landmarks for the detected face
            shape = predictor(gray, face)
            shape = [(p.x, p.y) for p in shape.parts()]

            # Detect the pupils using the facial landmarks
            left_eye, right_eye = detect_pupils(frame, shape)

            # Accumulate pupil positions for stabilization
            pupil_history.append((left_eye, right_eye))
            if len(pupil_history) > num_frames_accumulated:
                pupil_history.pop(0)

            # Calculate the average pupil position
            avg_left_eye = np.mean(pupil_history, axis=0, dtype=int)[0]
            avg_right_eye = np.mean(pupil_history, axis=0, dtype=int)[1]

            # Calculate the eye aspect ratio for both eyes
            left_eye_landmarks = np.array(shape[36:42])
            right_eye_landmarks = np.array(shape[42:48])
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)

            # Check if the eyes are closed
            ear_threshold = 0.2
            if left_ear < ear_threshold or right_ear < ear_threshold:
                # Eyes are closed, skip drawing the rectangle boxes and red pupil marks
                continue

            # Draw circles around the averaged pupil positions
            cv2.circle(frame, tuple(avg_left_eye), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(avg_right_eye), 3, (0, 0, 255), -1)

            # Classify the position of the pupil
            displacement_threshold = 8  # Adjust this threshold as needed
            position = classify_pupil_position(avg_left_eye, avg_right_eye, displacement_threshold)
            if position != "Center":
                position_history.append(time.time())
            else:
                position_history.clear()

            # Check if the position has not been "Center" for a long time
            attention_time_threshold = 5  # Time threshold in seconds
            if len(position_history) > 0 and (time.time() - position_history[0]) > attention_time_threshold:
                # Display "Attention!" on the frame
                cv2.putText(frame, "Attention!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



            # Draw the position text on the frame
            cv2.putText(frame, f"Pupil Position: {position}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Pupil Position Detection', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
