import cv2
import numpy as np
import tensorflow as tf
from openpose import pyopenpose as op

# Load the trained model
model = tf.keras.models.load_model('pose_estimation_model.h5')

# Initialize OpenPose parameters
params = dict()
params["model_folder"] = "C:/openpose/models"
params["model_pose"] = "BODY_25"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Create a VideoCapture object to read video from a file or camera feed
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Use OpenPose to detect the keypoints of each person in the frame
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Pre-process the frame and extract the keypoints
        frame = cv2.resize(frame, (256, 256))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0
        keypoints = datum.poseKeypoints

        # Use the model to predict the pose
        prediction = model.predict(frame)

        # Check for specific poses associated with each stretch
        if keypoints is not None:
            # Leg Swings
            if keypoints[0][14][0] < keypoints[0][11][0]:
                print("Leg Swings")

            # Walking Lunges
            elif keypoints[0][12][1] > keypoints[0][9][1]:
                print("Walking Lunges")

            # High Knees
            elif keypoints[0][15][1] < keypoints[0][11][1]:
                print("High Knees")

            # Butt Kicks
            elif keypoints[0][16][1] > keypoints[0][8][1]:
                print("Butt Kicks")

            # Leg Crossovers
            elif keypoints[0][14][0] > keypoints[0][11][0] and keypoints[0][11][0] > keypoints[0][8][0]:
                print("Leg Crossovers")

            # Ankle Bounces
            elif keypoints[0][19][1] < keypoints[0][18][1]:
                print("Ankle Bounces")

        # Display the results
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()