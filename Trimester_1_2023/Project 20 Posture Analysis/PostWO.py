import time
import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.python.solutions.pose import PoseLandmark
import numpy as np
import mediapipe as mp
import Analyser

class PWRefImages:
    def __init__(self, image_path, keypoints):
        self.image_path = image_path
        self.keypoints = keypoints

class Timer:
    def __init__(self):
        self.start_time = None
        self.paused_time = 0
        self.paused = False

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        elif self.paused:
            self.start_time = time.time() - self.paused_time
            self.paused = False

    def pause(self):
        if not self.paused:
            self.paused_time = time.time() - self.start_time
            self.paused = True

    def time_elapsed(self):
        if self.paused:
            return self.paused_time
        elif self.start_time is None:
            return 0
        else:
            return time.time() - self.start_time

    def reset(self):
        self.start_time = None
        self.paused_time = 0
        self.paused = False

class PostWorkoutClass:
    def __init__(self):
        self.cap = cv2.VideoCapture(3)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.IsComplete = False
        self.timer = Timer() 
        

    def preprocess_reference_image(image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(image_rgb)
            reference_landmarks = results.pose_landmarks

        return reference_landmarks

    def calculate_reference_angles(reference_landmarks, keypoints):
        reference_angles = []
        PA=Analyser.PoseAnalyzer()

        for keypoint_set in keypoints:
            K1=[reference_landmarks.landmark[keypoint_set[0]].x,reference_landmarks.landmark[keypoint_set[0]].y]
            K2=[reference_landmarks.landmark[keypoint_set[1]].x,reference_landmarks.landmark[keypoint_set[1]].y]
            K3=[reference_landmarks.landmark[keypoint_set[2]].x,reference_landmarks.landmark[keypoint_set[2]].y]
            angle = PA.calculate_angle(K1,K2,K3)
            reference_angles.append(angle)

        return reference_angles

    reference_images = [
        PWRefImages('PostWorkout Images/ChildsPose.jpg', [
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value),
            (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value)
        ]),
        PWRefImages('PostWorkout Images/ForwardFold.jpg', [
            (PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value),
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value)]),
        PWRefImages('PostWorkout Images/HamstringStretch.jpg', [
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value),
            ( PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value,PoseLandmark.RIGHT_INDEX.value),
            (PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_KNEE.value, PoseLandmark.LEFT_ANKLE.value)]),
        PWRefImages('PostWorkout Images/QuadStretch.jpg', [
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value),
            (PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_KNEE.value, PoseLandmark.LEFT_ANKLE.value)]),
        PWRefImages('PostWorkout Images/ShoulderStretch.jpg', [
            (PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_WRIST.value)]),
    ]

    reference_landmarks_list = []
    reference_angles_list = []

    for ref_image in reference_images:
        ref_landmarks = preprocess_reference_image(ref_image.image_path)
        ref_angles = calculate_reference_angles(ref_landmarks, ref_image.keypoints)

        reference_landmarks_list.append(ref_landmarks)
        reference_angles_list.append(ref_angles)


    
    def analyze_landmarks(self, image):
        # Analyze and display key landmarks
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image, results

    def display_landmarks(self, image, results, is_correct_form):
        if results.pose_landmarks:
            if is_correct_form:
                landmark_drawing_spec = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
            else:
                landmark_drawing_spec = DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)

            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=landmark_drawing_spec,
                                           connection_drawing_spec=landmark_drawing_spec)


    def childs_pose_analysis(self, landmarks, reference_angles):
        PA=Analyser.PoseAnalyzer()
        print("analysing childspose")

        # Calculate the user's angles
        user_hip_to_heel_angle = PA.calculate_angle(landmarks[PoseLandmark.LEFT_HIP.value],
                                                 landmarks[PoseLandmark.LEFT_KNEE.value],
                                                 landmarks[PoseLandmark.LEFT_ANKLE.value])
        user_shoulder_to_knee_angle = PA.calculate_angle(landmarks[PoseLandmark.LEFT_SHOULDER.value],
                                                      landmarks[PoseLandmark.LEFT_HIP.value],
                                                      landmarks[PoseLandmark.LEFT_KNEE.value])

        # Set a tolerance for angle comparisons
        tolerance = 10
        print (user_shoulder_to_knee_angle)

        # Compare the user's angles with the reference angles
        if (abs(user_hip_to_heel_angle - reference_angles[0]) <= tolerance and
            abs(user_shoulder_to_knee_angle - reference_angles[1]) <= tolerance):
            return True
        else:
            return False

    def childs_pose(self):
        self.IsComplete = False
        reference_angles = self.reference_angles_list[0]  # Assuming Child's Pose is the first reference image
        is_correct_form = False 
        while not self.IsComplete:
            
            _, frame = self.cap.read()
            image, results = self.analyze_landmarks(frame)

            if results.pose_landmarks:
                is_correct_form = self.childs_pose_analysis(results.pose_landmarks, reference_angles)

                if is_correct_form:
                    # Start the timer
                    self.timer.start()

                else:
                    # Pause the timer
                    self.timer.pause()
            print(self.timer.time_elapsed())
            if self.timer.time_elapsed() >= 30:
                self.IsComplete = True
                self.timer.reset()

            self.display_landmarks(image, results, is_correct_form)
            cv2.imshow('Post Workout Stretching', image)
            if cv2.waitKey(5) & 0xFF ==  ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()   
            
    
    def forward_fold_analysis(self, landmarks, reference_angles):
        PA = Analyser.PoseAnalyzer()

        # Calculate the user's angles
        user_shoulder_to_knee_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[PoseLandmark.RIGHT_KNEE.value])
        user_hip_to_heel_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[PoseLandmark.RIGHT_KNEE.value],
                                                    landmarks[PoseLandmark.RIGHT_ANKLE.value])

        # Set a tolerance for angle comparisons
        tolerance = 10

        # Compare the user's angles with the reference angles
        if (abs(user_shoulder_to_knee_angle - reference_angles[0]) <= tolerance and
            abs(user_hip_to_heel_angle - reference_angles[1]) <= tolerance):
            return True
        else:
            return False

    def forward_fold(self):
        self.IsComplete = False
        reference_angles = self.reference_angles_list[1]  # Assuming Forward Fold is the second reference image
        is_correct_form = False
        while not self.IsComplete:
            _, frame = self.cap.read()
            image, results = self.analyze_landmarks(frame)

            if results.pose_landmarks:
                is_correct_form = self.forward_fold_analysis(results.pose_landmarks, reference_angles)

                if is_correct_form:
                    self.timer.start()
                else:
                    self.timer.pause()

            if self.timer.time_elapsed() >= 30:
                self.IsComplete = True
                self.timer.reset()

            self.display_landmarks(image, results, is_correct_form)
            cv2.imshow('Post Workout Stretching', image)
            if cv2.waitKey(5) & 0xFF ==  ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()  

    def hamstring_stretch_analysis(self, landmarks, reference_angles, side):
        PA = Analyser.PoseAnalyzer()

        if side == "right":
            # Calculate the user's angles for the right side
            user_hip_to_ankle_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[PoseLandmark.RIGHT_KNEE.value],
                                                        landmarks[PoseLandmark.RIGHT_ANKLE.value])
            user_knee_to_index_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_KNEE.value],
                                                        landmarks[PoseLandmark.RIGHT_ANKLE.value],
                                                        landmarks[PoseLandmark.RIGHT_INDEX.value])
            # Compare the user's angles with the reference angles
            if abs(user_hip_to_ankle_angle - reference_angles[1]) <= 10 and abs(user_knee_to_index_angle - reference_angles[2]) <= 10:
                return True
            else:
                return False
        elif side == "left":
            # Calculate the user's angles for the left side
            user_hip_to_ankle_angle = PA.calculate_angle(landmarks[PoseLandmark.LEFT_HIP.value],
                                                        landmarks[PoseLandmark.LEFT_KNEE.value],
                                                        landmarks[PoseLandmark.LEFT_ANKLE.value])
            # Compare the user's angles with the reference angles
            if abs(user_hip_to_ankle_angle - reference_angles[0]) <= 10:
                return True
            else:
                return False
    def hamstring_stretch(self, side):
        self.IsComplete = False
        reference_angles = self.reference_angles_list[2]  # Assuming Hamstring Stretch is the third reference image
        is_correct_form = False 
        while not self.IsComplete:
            _, frame = self.cap.read()
            image, results = self.analyze_landmarks(frame)

            if results.pose_landmarks:
                is_correct_form = self.hamstring_stretch_analysis(results.pose_landmarks, reference_angles, side)

                if is_correct_form:
                    # Start the timer
                    self.timer.start()
                else:
                    # Pause the timer
                    self.timer.pause()

            if self.timer.time_elapsed() >= 30:
                self.IsComplete = True
                self.timer.reset()

            self.display_landmarks(image, results, is_correct_form)
            cv2.imshow('Post Workout Stretching', image)
            if cv2.waitKey(5) & 0xFF ==  ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
            
    def quad_stretch_analysis(self, landmarks, side, reference_angles):
        PA = Analyser.PoseAnalyzer()

        # Check which side is being stretched and adjust the keypoints accordingly
        if side == "right":
            user_stretched_hip_to_knee_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_HIP.value],
                                                                landmarks[PoseLandmark.RIGHT_KNEE.value],
                                                                landmarks[PoseLandmark.RIGHT_ANKLE.value])
            user_non_stretched_hip_to_knee_angle = PA.calculate_angle(landmarks[PoseLandmark.LEFT_HIP.value],
                                                                    landmarks[PoseLandmark.LEFT_KNEE.value],
                                                                    landmarks[PoseLandmark.LEFT_ANKLE.value])
        else: # side == "left"
            user_stretched_hip_to_knee_angle = PA.calculate_angle(landmarks[PoseLandmark.LEFT_HIP.value],
                                                                landmarks[PoseLandmark.LEFT_KNEE.value],
                                                                landmarks[PoseLandmark.LEFT_ANKLE.value])
            user_non_stretched_hip_to_knee_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_HIP.value],
                                                                    landmarks[PoseLandmark.RIGHT_KNEE.value],
                                                                    landmarks[PoseLandmark.RIGHT_ANKLE.value])

        # Set a tolerance for angle comparisons
        tolerance = 10

        # Compare the user's angles with the reference angles
        if (abs(user_stretched_hip_to_knee_angle - reference_angles[1]) <= tolerance and
            abs(user_non_stretched_hip_to_knee_angle - reference_angles[0]) <= tolerance):
            return True
        else:
            return False
    def quad_stretch(self, side):
        self.IsComplete = False
        reference_angles = self.reference_angles_list[3]  # Assuming Quad Stretch is the fourth reference image
        is_correct_form = False 
        while not self.IsComplete:
            _, frame = self.cap.read()
            image, results = self.analyze_landmarks(frame)

            if results.pose_landmarks:
                is_correct_form = self.quad_stretch_analysis(results.pose_landmarks, side, reference_angles)

                if is_correct_form:
                    # Start the timer
                    self.timer.start()
                else:
                    # Pause the timer
                    self.timer.pause()

            if self.timer.time_elapsed() >= 30:
                self.IsComplete = True
                self.timer.reset()

            self.display_landmarks(image, results, is_correct_form)
            cv2.imshow('Post Workout Stretching', image)
            if cv2.waitKey(5) & 0xFF ==  ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()            

    def shoulder_chest_stretch_analysis(self, landmarks, reference_angles):
        PA = Analyser.PoseAnalyzer()

        # Calculate the user's angles
        user_shoulder_to_wrist_angle = PA.calculate_angle(landmarks[PoseLandmark.RIGHT_HIP.value],
                                                        landmarks[PoseLandmark.RIGHT_SHOULDER.value],
                                                        landmarks[PoseLandmark.RIGHT_WRIST.value])

        # Set a tolerance for angle comparisons
        tolerance = 10

        # Compare the user's angles with the reference angles
        if abs(user_shoulder_to_wrist_angle - reference_angles[0]) <= tolerance:
            return True
        else:
            return False
        
    def shoulder_chest_stretch(self):
        self.IsComplete = False
        reference_angles = self.reference_angles_list[4]  # Assuming Shoulder Chest Stretch is the fifth reference image
        is_correct_form = False 
        while not self.IsComplete:
            _, frame = self.cap.read()
            image, results = self.analyze_landmarks(frame)

            if results.pose_landmarks:
                is_correct_form = self.shoulder_chest_stretch_analysis(results.pose_landmarks, reference_angles)

                if is_correct_form:
                    # Start the timer
                    self.timer.start()
                else:
                    # Pause the timer
                    self.timer.pause()

            if self.timer.time_elapsed() >= 30:
                self.IsComplete = True
                self.timer.reset()

            self.display_landmarks(image, results, is_correct_form)
            cv2.imshow('Post Workout Stretching', image)
            if cv2.waitKey(5) & 0xFF ==  ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def perform_stretches(self):
        # Perform symmetrical stretches
        self.forward_fold()
        time.sleep(10)
        self.shoulder_chest_stretch()
        time.sleep(10)
        self.childs_pose()
        time.sleep(10)

        # Perform asymmetrical stretches
        self.quad_stretch("right")
        time.sleep(10)
        self.quad_stretch("left")
        time.sleep(10)
        self.hamstring_stretch("right")
        time.sleep(10)
        self.hamstring_stretch("left")
        time.sleep(10)


        print("Congratulations! You have completed all the stretches.")

  
if __name__ == "__main__":
    post_workout = PostWorkoutClass()
    post_workout.perform_stretches()