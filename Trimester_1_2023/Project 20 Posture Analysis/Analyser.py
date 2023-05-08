import cv2
import mediapipe as mp
import numpy as np
import json
import datetime
import time

class PoseAnalyzer:

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.quad_stretch_start_time = None

        self.greenzone = (51, 212, 75)
        self.redzone = (75, 51, 212)
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
        pass
    def quad_stretch(self):
        # Check if quad stretch is already in progress
        if self.quad_stretch_start_time is not None:
            current_time = time.time()
            elapsed_time = current_time - self.quad_stretch_start_time
            
            # Check if 30 seconds have passed
            if elapsed_time >= 30:
                # Reset the timer
                self.quad_stretch_start_time = None
                return  # Exit the method
            
            # Check the quad angle
            hip = self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = self.landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = self.landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            quad_angle = self.calculate_angle(hip, knee, ankle)
            
            # Check if quad angle is within desired range (0-20 degrees)
            if quad_angle < 0 or quad_angle > 20:
                # Reset the timer
                self.quad_stretch_start_time = None
                return  # Exit the method
        
        # If quad stretch is not in progress or quad angle is within desired range,
        # start/restart the timer and perform the quad stretch
        self.quad_stretch_start_time = time.time()
    
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
        pass

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
    def calculate_angle(self, a, b, c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def get_non_absolute(self, a, b, c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        return angle

    def analyze_posture(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = self.pose.process(image)

            image.flags.writeable = True
  
            try:
                landmarks = results.pose_landmarks

                # Analyze body position
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Check for elbow angle
                elbowangle = self.calculate_angle(shoulder, elbow, wrist)

                if 145 < elbowangle < 165:
                    cv2.putText(image, str(round(elbowangle)), 
                               tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.greenzone, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image,(str(round(elbowangle)),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.redzone, 2, cv2.LINE_AA))
                    # Check for hip angle
                    hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    hipangle = self.calculate_angle(shoulder, hip, knee)

                #check for hip angle
                if 50 < hipangle < 75:
                    cv2.putText(image, str(round(hipangle)), 
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.greenzone, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, str(round(hipangle)), 
                                tuple(np.multiply(hip, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.redzone, 2, cv2.LINE_AA)

                # Check for wrist flex
                knuckles = [landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value].y]

                wristangle = self.calculate_angle(elbow, wrist, knuckles)

                if wristangle > 170:
                    cv2.putText(image, str(round(wristangle)), 
                                tuple(np.multiply(wrist, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.greenzone, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, str(round(wristangle)), 
                                tuple(np.multiply(wrist, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.redzone, 2, cv2.LINE_AA)

                # Check for knee extension
                heel = [landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].y]

                kneeangle = self.calculate_angle(hip, knee, heel)

                if 125 < kneeangle < 155:
                    cv2.putText(image, str(round(kneeangle)), 
                                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.greenzone, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, str(round(kneeangle)), 
                                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.redzone, 2, cv2.LINE_AA)

                #if the left heel is ahead of the right heel were in a downstroke
                rightheel=[landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                leftheel=heel
                lefttoe=[landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                pedalpoint=[landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                
                if leftheel[0]>rightheel[0]:
                    downstroke=True
            
                elif leftheel[0]<rightheel[0]:
                    downstroke=False
                
                
                ballangle=self.get_non_absolute(leftheel,lefttoe,pedalpoint)
                
                    
            #     # Analyze pedalling technique
                if not downstroke:
                    if ballangle >15  and ballangle<25:
                            cv2.putText(image, str(round(ballangle)), 
                            tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (self.greenzone), 2, cv2.LINE_AA)
                    
                    else:
                            cv2.putText(image, str(round(ballangle)), 
                            tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (self.redzone), 2, cv2.LINE_AA)
                    
                else:
                    if ballangle <5 and ballangle>15:
                            cv2.putText(image, str(round(ballangle)), 
                            tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (self.greenzone), 2, cv2.LINE_AA)
                    else:
                            cv2.putText(image, str(round(ballangle)), 
                            tuple(np.multiply(leftheel, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (self.redzone), 2, cv2.LINE_AA)
                     
                  #Analyse Aero
               # Check for shoulder and forearm position (aero check)
               
                forearm_parallel = abs(wrist[1] - elbow[1]) <= 0.05  # Check if forearm is parallel to the ground
                if forearm_parallel and 25 <= hipangle <= 35:
                    cv2.putText(image, "Aero Efficient", tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.greenzone, 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Not Aero Efficient", tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.redzone, 2, cv2.LINE_AA)
                                                    
            except:
                pass

            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(127, 79, 209), thickness=2, circle_radius=2),
                                        self.mp_drawing.DrawingSpec(color=(213, 161, 54), thickness=2, circle_radius=2))

            cv2.imshow('Posture Analysis', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def run(self):
    self.analyze_posture()   