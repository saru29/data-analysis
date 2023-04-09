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
cap = cv2.VideoCapture(1)


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
            
            greenzone=(51,212,75)
            redzone=(75,51,212)
            landmarks=results.pose_landmarks.landmark
        
            # Analyze body position
            
            # Get relevant landmarks for cycling posture detection --take their left  cross-section only
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
           
            # Check for elbow angle
            elbowangle= calculate_angle(shoulder, elbow, wrist)
     
            
            if elbowangle>145 and elbowangle<165:
                cv2.putText(image, str(round(elbowangle)), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, greenzone, 2, cv2.LINE_AA
                                )
            else:
                cv2.putText(image, str(round(elbowangle)), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA
                                )               
        
            # Check for hip angle
            hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            hipangle=calculate_angle(shoulder,hip,knee)
            
            if hipangle<75 and hipangle>50:
                cv2.putText(image, str(round(hipangle)), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (greenzone), 2, cv2.LINE_AA)    
            else:
                  cv2.putText(image, str(round(hipangle)), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA)
                   
                
            #check for wrist flex
            knuckles = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
       
            wristangle= calculate_angle(elbow, wrist, knuckles)
            
            if wristangle>170 :
                cv2.putText(image, str(round(wristangle)), 
                            tuple(np.multiply(wrist, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (greenzone), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str(round(wristangle)), 
                        tuple(np.multiply(wrist, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA)
                
            #check for knee extension
            heel=[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            
            kneeangle=calculate_angle(hip,knee,heel)
            
            
            #add an up/downstroke check here
            if kneeangle >125 and kneeangle<155:
                cv2.putText(image, str(round(kneeangle)), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (greenzone), 2, cv2.LINE_AA)
                
            else:
                cv2.putText(image, str(round(kneeangle)), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA)
                                   

        except:
            pass


        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(127,79,209),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(213,161,54),thickness=2,circle_radius=2))
        cv2.imshow('Video Feed',image)


         #if the left heel is ahead of the right heel were in an upstroke
        rightheel=heel=[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

    #     # Analyze pedalling technique
        if stroke!='down':
            if ballangle >15  and ballangle<25:
                #greenzone
            else:
                #redzone
        else:
            if ballangle <-5 and ballangle>-15:
                #greenzone
            else:
                #redzone
                
                
    #     # Analyze aerodynamics
           ##contour detection+transluscent image overlay+pose estimation


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
