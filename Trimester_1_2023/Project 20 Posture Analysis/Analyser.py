import cv2
import mediapipe as mp
import numpy as np
import json
import datetime
import random

class PoseAnalyzer:


    def __init__(self):
        self.cap = cv2.VideoCapture("Bike.mp4")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.workout_data = []
        self.is_recording = False
        self.record_start_time = None
        self.record_limit = 60 * 10  # e.g., 10 minutes
        

    
    def start_recording(self):
        self.is_recording = True
        self.record_start_time = datetime.datetime.now()
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.record_file_name = f"Workouts\Videos\workout_{current_time}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.record_file_name, fourcc, 15, (640, 480))


    def stop_recording(self):
        self.is_recording = False
        self.record_start_time = None
        self.out.release()
        self.out = None
        


       
            
        
    def save_workout_data(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"Workouts\Data\workout_{current_time}.json"
        with open(file_name, "w") as f:
            json.dump(self.workout_data, f)
            
    def load_dataset(self,file_path):
        with open(file_path, 'r') as file:          
            dataset =json.load(file)
        return dataset
    


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


    def calculate_dataset_angles(self, dataset_images):
        dataset_joint_angles = []
        for image_path in dataset_images:
            image = cv2.imread(image_path)
            if image is not None:
                results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                keypoints = results.pose_landmarks

                if keypoints is not None:
                    key_landmarks = [
                        self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                        self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                        self.mp_pose.PoseLandmark.LEFT_WRIST.value,
                        self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
                        self.mp_pose.PoseLandmark.LEFT_HIP.value,
                        self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                        self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                        self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                        self.mp_pose.PoseLandmark.LEFT_HEEL.value,
                        self.mp_pose.PoseLandmark.RIGHT_HEEL.value,
                        self.mp_pose.PoseLandmark.LEFT_INDEX.value,
                        self.mp_pose.PoseLandmark.RIGHT_INDEX.value,
                    ]
                 #elbow angle
                    if (
                        (keypoints.landmark[key_landmarks[0]].visibility > 0 and
                        keypoints.landmark[key_landmarks[2]].visibility > 0 and
                        keypoints.landmark[key_landmarks[4]].visibility > 0) or
                        (keypoints.landmark[key_landmarks[1]].visibility > 0 and
                        keypoints.landmark[key_landmarks[3]].visibility > 0 and
                        keypoints.landmark[key_landmarks[5]].visibility > 0)
                    ):
                        if (
                            keypoints.landmark[key_landmarks[0]].visibility +
                            keypoints.landmark[key_landmarks[2]].visibility +
                            keypoints.landmark[key_landmarks[4]].visibility >
                            keypoints.landmark[key_landmarks[1]].visibility +
                            keypoints.landmark[key_landmarks[3]].visibility +
                            keypoints.landmark[key_landmarks[5]].visibility
                        ):
                            landmarks = [
                                keypoints.landmark[key_landmarks[0]],
                                keypoints.landmark[key_landmarks[2]],
                                keypoints.landmark[key_landmarks[4]]
                            ]
                        else:
                            landmarks = [
                                keypoints.landmark[key_landmarks[1]],
                                keypoints.landmark[key_landmarks[3]],
                                keypoints.landmark[key_landmarks[5]]
                            ]

                        elbow_angle =self.calculate_angle([landmarks[0].x,landmarks[0].y], [landmarks[1].x,landmarks[1].y], [landmarks[2].x,landmarks[2].y])
                       
                        
                    else:
                
                      elbow_angle=random.uniform(150, 160)  # Append random bound angle
                      
                     #hip angle 
                    if (
                        (keypoints.landmark[key_landmarks[0]].visibility > 0 and
                        keypoints.landmark[key_landmarks[6]].visibility > 0 and
                        keypoints.landmark[key_landmarks[8]].visibility > 0) or
                        (keypoints.landmark[key_landmarks[1]].visibility > 0 and
                        keypoints.landmark[key_landmarks[7]].visibility > 0 and
                        keypoints.landmark[key_landmarks[9]].visibility > 0)
                    ):
                        if (
                            keypoints.landmark[key_landmarks[0]].visibility +
                            keypoints.landmark[key_landmarks[6]].visibility +
                            keypoints.landmark[key_landmarks[8]].visibility >
                            keypoints.landmark[key_landmarks[1]].visibility +
                            keypoints.landmark[key_landmarks[7]].visibility +
                            keypoints.landmark[key_landmarks[9]].visibility
                        ):
                            landmarks = [
                                keypoints.landmark[key_landmarks[0]],
                                keypoints.landmark[key_landmarks[6]],
                                keypoints.landmark[key_landmarks[8]]
                            ]
                        else:
                            landmarks = [
                                keypoints.landmark[key_landmarks[1]],
                                keypoints.landmark[key_landmarks[7]],
                                keypoints.landmark[key_landmarks[9]]
                            ]

                        hip_angle = self.calculate_angle([landmarks[0].x,landmarks[0].y], [landmarks[1].x,landmarks[1].y], [landmarks[2].x,landmarks[2].y])
                        
                    else:
                        hip_angle=random.uniform(55, 70)
                    #wrist angle
                    if (
                        (keypoints.landmark[key_landmarks[2]].visibility > 0 and
                        keypoints.landmark[key_landmarks[4]].visibility > 0 and
                        keypoints.landmark[key_landmarks[12]].visibility > 0) or
                        (keypoints.landmark[key_landmarks[3]].visibility > 0 and
                        keypoints.landmark[key_landmarks[5]].visibility > 0 and
                        keypoints.landmark[key_landmarks[13]].visibility > 0)
                    ):
                        if (
                            keypoints.landmark[key_landmarks[2]].visibility +
                            keypoints.landmark[key_landmarks[4]].visibility +
                            keypoints.landmark[key_landmarks[12]].visibility >
                            keypoints.landmark[key_landmarks[3]].visibility +
                            keypoints.landmark[key_landmarks[5]].visibility +
                            keypoints.landmark[key_landmarks[13]].visibility
                        ):
                            landmarks = [
                                keypoints.landmark[key_landmarks[2]],
                                keypoints.landmark[key_landmarks[4]],
                                keypoints.landmark[key_landmarks[12]]
                            ]
                        else:
                            landmarks = [
                                keypoints.landmark[key_landmarks[3]],
                                keypoints.landmark[key_landmarks[5]],
                                keypoints.landmark[key_landmarks[13]]
                            ]

                        wrist_angle = self.calculate_angle([landmarks[0].x,landmarks[0].y], [landmarks[1].x,landmarks[1].y], [landmarks[2].x,landmarks[2].y])
                       
                    else:
                        wrist_angle=random.uniform(170, 180)
                    #knee angle
                    if (
                        (keypoints.landmark[key_landmarks[6]].visibility > 0 and
                        keypoints.landmark[key_landmarks[8]].visibility > 0 and
                        keypoints.landmark[key_landmarks[10]].visibility > 0) or
                        (keypoints.landmark[key_landmarks[7]].visibility > 0 and
                        keypoints.landmark[key_landmarks[9]].visibility > 0 and
                        keypoints.landmark[key_landmarks[11]].visibility > 0)
                    ):
                        if (
                            keypoints.landmark[key_landmarks[6]].visibility +
                            keypoints.landmark[key_landmarks[8]].visibility +
                            keypoints.landmark[key_landmarks[10]].visibility >
                            keypoints.landmark[key_landmarks[7]].visibility +
                            keypoints.landmark[key_landmarks[9]].visibility +
                            keypoints.landmark[key_landmarks[11]].visibility
                        ):
                            landmarks = [
                                keypoints.landmark[key_landmarks[6]],
                                keypoints.landmark[key_landmarks[8]],
                                keypoints.landmark[key_landmarks[10]]
                            ]
                        else:
                            landmarks = [
                                keypoints.landmark[key_landmarks[7]],
                                keypoints.landmark[key_landmarks[9]],
                                keypoints.landmark[key_landmarks[11]]
                            ]

                        knee_angle = self.calculate_angle([landmarks[0].x,landmarks[0].y], [landmarks[1].x,landmarks[1].y], [landmarks[2].x,landmarks[2].y])
                        
                    else:
                        knee_angle=random.uniform(130, 150)
                        
                    dataset_joint_angles.append((hip_angle, knee_angle, elbow_angle,wrist_angle))
                    
                else:
                       dataset_joint_angles.append((
                        random.uniform(55, 70),   # Default value for hip angle
                        random.uniform(130, 150),  # Default value for knee angle
                        random.uniform(150, 160),  # Default value for elbow angle
                        random.uniform(170, 180))  # Default value for wrist angle
                    )
                    
            else:
                print(f"Failed to load image: {image_path}")
                

        return dataset_joint_angles
 

    def replace_outliers(self,dataset, zscore_threshold):
        # Convert the dataset to a NumPy array for easier manipulation
        dataset_array = np.array(dataset)

        # Calculate the z-scores for each data point
        zscores = (dataset_array - np.mean(dataset_array)) / np.std(dataset_array)

        # Find the indices of outliers
        outlier_indices = np.where(np.abs(zscores) > zscore_threshold)[0]

        # Remove outliers from the dataset
        non_outlier_dataset = np.delete(dataset_array, outlier_indices)

        # Calculate the replacement range using the non-outlying data
        range_start = np.min(non_outlier_dataset)
        range_end = np.max(non_outlier_dataset)

        # Replace outliers with random values within the calculated range
        for idx in outlier_indices:
            dataset_array[idx] = random.uniform(range_start, range_end)

        # Convert the dataset_array back to a list and return
        return dataset_array.tolist()


    def record_frame(self,frame):
       if self.is_recording:
        resized_frame = cv2.resize(frame, (640, 480))
        
        self.out.write(resized_frame)

    def analyze_posture(self,dataset_joint_angles,similarity_threshold=0.01):
      
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            hipState=False
            kneeState=False
            elbowState=False
            wristState=False
            pedalState=False
            AeroState=False
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = self.pose.process(image)

            image.flags.writeable = True
            # Define the threshold for considering values as outliers
            zscore_threshold = 3.0  # Adjust as needed

            joint_angles=self.replace_outliers(dataset_joint_angles,zscore_threshold)
                            #save the state of the cyclist
 
            try:
                landmarks = results.pose_landmarks
                
                # Calculate the joint angles for the live video frame
                shoulder = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                heel = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                knuckles = [landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                
                
                hip_angle = self.calculate_angle(shoulder, hip, knee)
                knee_angle = self.calculate_angle(hip, knee, heel)
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                wrist_angle=self.calculate_angle(elbow,wrist,knuckles)

                
                
              
                # Compare the joint angles with the dataset
               
                for dataset_angles in joint_angles:
                    hip_angle_dataset, knee_angle_dataset, elbow_angle_dataset, wrist_angle_dataset = dataset_angles

                    hip_angle_diff = abs(hip_angle - hip_angle_dataset)
                    knee_angle_diff = abs(knee_angle - knee_angle_dataset)
                    elbow_angle_diff = abs(elbow_angle - elbow_angle_dataset)
                    wrist_angle_diff = abs(wrist_angle - wrist_angle_dataset)
                    
                    
                    
                    # Define the threshold range for each joint angle
                    hip_angle_threshold = 80  # Adjust as needed
                    knee_angle_threshold = 60  # Adjust as needed
                    elbow_angle_threshold = 60  # Adjust as needed
                    wrist_angle_threshold = 30  # Adjust as needed

                    # Calculate the similarity scores for each joint angle
                    hip_similarity_score = 1 - (hip_angle_diff / hip_angle_threshold)
                    knee_similarity_score = 1 - (knee_angle_diff / knee_angle_threshold)
                    elbow_similarity_score = 1 - (elbow_angle_diff / elbow_angle_threshold)
                    wrist_similarity_score = 1 - (wrist_angle_diff / wrist_angle_threshold)
                    
                    hipState = True if hip_similarity_score > similarity_threshold else False
                    kneeState = True if knee_similarity_score > similarity_threshold else False
                    elbowState = True if elbow_similarity_score > similarity_threshold else False
                    wristState = True if wrist_similarity_score > similarity_threshold else False
                    # Determine the overall similarity score based on the average of individual scores
                    similarity_score = (hip_similarity_score + knee_similarity_score + elbow_similarity_score + wrist_similarity_score) / 4
                    similarity_score = max(similarity_score, 0)  # Ensure similarity score is not negative

                    
               
                toe=[landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                pedalpoint=[landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]    
                
                #identify at what pedalling point we are
                if toe[0]>knee[0]:
                    downstroke=True
            
                elif toe[0]<knee[0]:
                    downstroke=False

         
                ballangle=self.get_non_absolute(heel,toe,pedalpoint)                        
                # Analyze pedalling technique
                if not downstroke:
                    if ballangle > 0 and ballangle < 45:
                        pedalState=True
                    else:
                       pedalState=False
                elif downstroke:
                    if ballangle < 0 :
                        pedalState=True
                    else:
                       pedalState=False

                nose = [landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].x, landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].y]
                head_position = self.calculate_angle(shoulder, hip, nose)
                body_angle = self.calculate_angle(hip, shoulder, [shoulder[0], hip[1]]) 
                
                # Analyze Aero
                aerodynamics_score = (body_angle + elbow_angle + head_position + (180 - knee_angle)) / 4
                if aerodynamics_score>50:
                   AeroState=True
                else:
                   AeroState=False
                   
                   
                instructions = "Press 'r' to record this current session"
                cv2.putText(image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                    
                    
                
                if self.is_recording:
                    now = datetime.datetime.now()
           
                    elapsed_time = (now - self.record_start_time).total_seconds()
                    elapsed_time_str = f"Recording time: {elapsed_time}s"
                    cv2.putText(image, elapsed_time_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                    if elapsed_time > self.record_limit:
                        self.stop_recording()

                if self.is_recording:
                    # Populate workout_data
                    data_point = {
                        'timestamp': now.strftime("%Y%m%d%H%M%S"),
                        'hip_similarity_score': hip_similarity_score,
                        'knee_similarity_score': knee_similarity_score,
                        'elbow_similarity_score': elbow_similarity_score,
                        'wrist_similarity_score': wrist_similarity_score,
                        'hip_state': hipState,
                        'knee_state': kneeState,
                        'elbow_state': elbowState,
                        'wrist_state': wristState,
                        'downstroke': downstroke,
                        'pedal_state': pedalState,
                        'aero_state': AeroState,
                        'aerodynamics_score': aerodynamics_score
                    }
                    self.workout_data.append(data_point)
                                      
            except:
                pass



            true_color = (0, 255, 0)  # Green color
            false_color = (255, 0,0 )  # Red color
            neutral=(255,255,255)
            try:
                if landmarks is not None: 
                    # Convert the joint coordinates back to the original image resolution
                    shoulderc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image.shape[0])]
                    elbowc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image.shape[0])]
                    wristc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0])]
                    hipc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * image.shape[0])]
                    kneec = [int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image.shape[0])]
                    heelc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HEEL.value].y * image.shape[0])]
                    rshoulderc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])]
                    relbowc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])]
                    rwristc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])]
                    rhipc = [int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1]), int(landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0])]
                  
            except:
                pass
            # ...
          
               
         
          
            # Draw circles at each joint
          
            if AeroState:
             cv2.circle(image, tuple(shoulderc), 5, true_color, -1)
             cv2.circle(image, tuple(rshoulderc), 5, true_color, -1)
            else:
             cv2.circle(image, tuple(shoulderc), 5, false_color, -1)
             cv2.circle(image, tuple(rshoulderc), 5, false_color, -1)
              
            if elbowState:
             cv2.circle(image, tuple(elbowc), 5, true_color, -1)
             cv2.circle(image, tuple(relbowc), 5, true_color, -1)
            else:
             cv2.circle(image, tuple(elbowc), 5, false_color, -1)
             cv2.circle(image, tuple(relbowc), 5, false_color, -1)
             
            if wristState:
             cv2.circle(image, tuple(wristc), 5, true_color, -1)
             cv2.circle(image, tuple(rwristc), 5, true_color, -1)
            else:
             cv2.circle(image, tuple(wristc), 5, false_color, -1)
             cv2.circle(image, tuple(rwristc), 5, false_color, -1)

            if hipState:
             cv2.circle(image, tuple(hipc), 5, true_color, -1)
             cv2.circle(image, tuple(rhipc), 5, true_color, -1)
            else:
             cv2.circle(image, tuple(hipc), 5, false_color, -1)
             cv2.circle(image, tuple(rhipc), 5, false_color, -1)

            if pedalState:
             cv2.circle(image, tuple(heelc), 5, true_color, -1)
            else:
             cv2.circle(image, tuple(heelc), 5, false_color, -1)
             
            if kneeState:
             cv2.circle(image, tuple(kneec), 5, true_color, -1)
            else:
             cv2.circle(image, tuple(kneec), 5, false_color, -1)
             

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                if self.is_recording:
                    self.stop_recording()
                    self.save_workout_data()
                else:
                    self.start_recording()
            

            output_image=image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.record_frame(output_image)
            cv2.imshow('Posture Analysis', output_image)
        

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self, dataset_joint_angles):
        self.analyze_posture(dataset_joint_angles, similarity_threshold=0.1) 

if __name__ == '__main__':
    pose_analyzer = PoseAnalyzer()
    
    # Load the dataset from the json file
    dataset_images = pose_analyzer.load_dataset('posturedataset.json')
    
    dataset_joint_angles = pose_analyzer.calculate_dataset_angles(dataset_images)
    pose_analyzer.run(dataset_joint_angles)