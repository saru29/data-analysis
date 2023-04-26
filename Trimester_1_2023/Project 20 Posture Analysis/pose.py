#Pre-Req: Can only use protobuf v3.2 not higher with the current mediapipe version

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp



# Initialize Mediapipe parameters
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a VideoCapture object to read video from a file or camera feed
#currently set at the available camera, later we can change to different cameras
cap = cv2.VideoCapture(2)


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
        try:
            landmarks=results.pose_landmarks.landmark
        
            # Analyze body position
            
            # Get relevant landmarks for cycling posture detection --take their left  cross-section only
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
           
            # Check for elbow angle
            elbowangle= calculate_angle(shoulder, elbow, wrist)

         
            
            # Visualize angle
            cv2.putText(image, str(elbowangle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,213, 240), 2, cv2.LINE_AA
                                )
            # Check for hip angle
            hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            hipangle=calculate_angle(shoulder,hip,knee)
        
            cv2.putText(image, str(hipangle), 
                           tuple(np.multiply(hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,213, 240), 2, cv2.LINE_AA)       
            
            #check for wrist flex
            knuckles = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
       
            wristangle= calculate_angle(elbow, wrist, knuckles)
            
            cv2.putText(image, str(wristangle), 
                           tuple(np.multiply(wrist, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,213, 240), 2, cv2.LINE_AA)
                 
            #check for knee extension
            heel=[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            
            kneeangle=calculate_angle(hip,knee,heel)

            cv2.putText(image, str(kneeangle), 
                           tuple(np.multiply(knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,213, 240), 2, cv2.LINE_AA)
            
            

        except:
            pass


        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(127,79,209),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(213,161,54),thickness=2,circle_radius=2))
        cv2.imshow('Video Feed',image)




    #     # Analyze pedalling technique

    #     # Analyze aerodynamics

    #     # Provide feedback
    #     feedback = ""
    #     if not hip_shoulders_level:
    #         feedback += "Your hips and shoulders are not level. Try to align your body to avoid asymmetries.\n"
    #     if not proper_ankling:
    #         feedback += "Your pedalling technique is not optimal. Try to use proper ankling to maximize power output and reduce wasted energy.\n"
    #     if frontal_surface_area > MAX_FRONTAL_SURFACE_AREA:
    #         feedback += "Your body position is not streamlined enough. Try to tuck in your elbows and lower your head to reduce drag.\n"
    #     if feedback == "":
    #         feedback = "You're doing great! Keep up the good work."

    #     # Display the results
    #     cv2.imshow('Pose Estimation', frame)
    #     print(feedback)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
