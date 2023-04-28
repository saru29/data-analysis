#Pre-Req: Can only use protobuf v3.2 not higher with the current mediapipe version

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import datetime


# Initialize Mediapipe parameters
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a VideoCapture object to read video from a file or camera feed
#currently set at the available camera, later we can change to different cameras
cap = cv2.VideoCapture(1)


#Allow recording
#fourcc = cv.VideoWriter_fourcc(*'MJPG')
#out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
workout_data = []
# Function to save workout data to a file
def save_workout_data(workout_data):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"workout_{current_time}.json"
    with open(file_name, "w") as f:
        json.dump(workout_data, f)
def perform_cooldown():
    # Add code here to perform cooldown exercises
    quad_stretch()
    forward_fold()
    spinal_twist()
    hamstring_stretch()
    child_pose()
    shoulder_chest_stretch()
def quad_stretch():
    # Add code here to perform quad stretch
    pass

def forward_fold():
    # Add code here to perform forward fold
    pass

def spinal_twist():
    # Add code here to perform spinal twist
    pass

def hamstring_stretch():
    # Add code here to perform hamstring stretch
    pass

def child_pose():
    # Add code here to perform child's pose
    pass

def shoulder_chest_stretch():
    # Add code here to perform shoulder and chest stretch
    pass
# Perform pre-workout stretching
def perform_pre_workout_stretching():
    # Cycling-oriented pre-stretches
    # Perform each stretch for the recommended duration
    hip_flexor_stretch()
    hamstring_stretch()
    quadriceps_stretch()
    calf_stretch()
    lower_back_stretch()
    upper_body_stretch()

def hip_flexor_stretch():
    # Add code here to perform hip flexor stretch
    pass

def hamstring_stretch():
    # Add code here to perform hamstring stretch
    pass

def quadriceps_stretch():
    # Add code here to perform quadriceps stretch
    pass

def calf_stretch():
    # Add code here to perform calf stretch
    pass

def lower_back_stretch():
    # Add code here to perform lower back stretch
    pass

def upper_body_stretch():
    # Add code here to perform upper body stretch
    pass
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def get_non_absolute(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    
        
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

         
        #recolor image to gray
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        # Load the Haar Cascade Classifier for human detection
        human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

        # Detect the humans in the image
        humans = human_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)

        print(humans)


        try:
            
            greenzone=(51,212,75)
            redzone=(75,51,212)
            landmarks=results.pose_landmaqrks.landmark
        
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
            
            
            
            if kneeangle >125 and kneeangle<155:
                cv2.putText(image, str(round(kneeangle)), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (greenzone), 2, cv2.LINE_AA)
                
            else:
                cv2.putText(image, str(round(kneeangle)), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA)
                                   
            #if the left heel is ahead of the right heel were in a downstroke
            rightheel=[landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            leftheel=heel
            lefttoe=[landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            pedalpoint=[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            
            
            if leftheel[0]>rightheel[0]:
             downstroke=True
           
            elif leftheel[0]<rightheel[0]:
             downstroke=False
             
             
            ballangle=get_non_absolute(leftheel,lefttoe,pedalpoint)
            
            
            
            
        #     # Analyze pedalling technique
            if not downstroke:
                if ballangle >15  and ballangle<25:
                        cv2.putText(image, str(round(ballangle)), 
                        tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (greenzone), 2, cv2.LINE_AA)
                
                else:
                        cv2.putText(image, str(round(ballangle)), 
                        tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA)
                
            else:
                if ballangle <5 and ballangle>15:
                        cv2.putText(image, str(round(ballangle)), 
                        tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (greenzone), 2, cv2.LINE_AA)
                else:
                        cv2.putText(image, str(round(ballangle)), 
                        tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (redzone), 2, cv2.LINE_AA)
                    
                    
        #     # Analyze aerodynamics
       
        # Draw the outlines around the human
            for (x, y, w, h) in humans:
              cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
          

            
                        


        except:
            pass


        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(127,79,209),thickness=2,circle_radius=2),
                                mp_drawing.DrawingSpec(color=(213,161,54),thickness=2,circle_radius=2))
        
        cv2.imshow('Posture Analysis',image)
        



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()