import cv2
import mediapipe as mp
import numpy as np
from Analyser import PoseAnalyzer
from Instruction import Instruction
from pydub import AudioSegment
from pydub.playback import play
import threading
from queue import Queue
import logging

logging.basicConfig(filename='debug.log', level=logging.DEBUG)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


        
class PoseComparison:
    def __init__(self, video_path, threshold):
        self.video_path = video_path
        self.threshold = threshold
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.analyser = PoseAnalyzer()
        self.instruction = Instruction(video_path)
        self.instruction.load_subtitles()
        self.desired_width = 800
      




   
            
    def calculate_and_display_angle(self, image, landmarks, joint1, joint2, joint3):
        # Get coordinates
        a = [landmarks[joint1.value].x, landmarks[joint1.value].y]
        b = [landmarks[joint2.value].x, landmarks[joint2.value].y]
        c = [landmarks[joint3.value].x, landmarks[joint3.value].y]
        
        # Calculate angle
        angle = self.analyser.calculate_angle(a, b, c)

        # Apply a threshold to check if the movement is correct
        if angle > self.threshold:  # define your threshold value
            color = (0, 255, 0)  # green
        else:
            color = (255, 0, 0)  # red

        # Draw the angle on the image
        cv2.circle(image, tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)), 5, color, 2)
        
    def resize_frame(self, frame):
        # Calculate the ratio of the new width to the old width
        ratio = self.desired_width / frame.shape[1]

        # Compute the new dimensions of the frame
        dim = (self.desired_width, int(frame.shape[0] * ratio))

        # Resize the frame
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        return resized_frame
    def compare(self):
        # Load videos
        cap = cv2.VideoCapture(self.video_path)
        cap_user = cv2.VideoCapture(self.video_path)  # user video feed      



            
        # Calculate the frame rate of the video
  
        frame_rate= cap.get(cv2.CAP_PROP_FPS)
        #maintain a constant frame rate on the video for an easier user following
        frame_skip_rate = int(frame_rate / 24)  # calculate how many frames to skip to maintain the desired fps

     
        frame_number = 0
        while cap.isOpened() and cap_user.isOpened():
            ret, frame = cap.read()
            ret_user, frame_user = cap_user.read()
            
            if not ret or not ret_user:
                logging.debug("Can't receive frame. Exiting ...")
                print("Can't receive frame. Exiting ...")
                break

            # Skip frames to maintain the desired frame rate
            if frame_number % frame_skip_rate != 0:
                frame_number += 1
                continue
            
            time = frame_number * (1000 / frame_rate) 
            # Mirror and resize the frames
            
            frame = self.resize_frame(frame)
            frame_user = self.resize_frame(frame_user)
            
            frame = cv2.flip(frame, 1)
            frame_user = cv2.flip(frame_user, 1)
            frame = self.display_instructions(frame, time)
            frame_number += 1
            # Recolor images to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)


            # Make detections
            results = self.pose.process(image)
            results_user = self.pose.process(image_user)
            
            # Render detections
            if results.pose_landmarks and results_user.pose_landmarks:
                for side in ['LEFT', 'RIGHT']:
                    for joint_set in [['HIP', 'SHOULDER', 'ELBOW'], ['WRIST', 'ELBOW', 'SHOULDER'], ['ELBOW', 'WRIST', 'INDEX'], ['SHOULDER', 'HIP', 'KNEE'], ['HIP', 'KNEE', 'ANKLE'], ['KNEE', 'ANKLE', 'HEEL']]:
                        self.calculate_and_display_angle(image, results.pose_landmarks.landmark, getattr(mp_pose.PoseLandmark, f'{side}_{joint_set[0]}'), getattr(mp_pose.PoseLandmark, f'{side}_{joint_set[1]}'), getattr(mp_pose.PoseLandmark, f'{side}_{joint_set[2]}'))
                
              
                
         
            # Convert the image back to BGR for rendering with OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Display the resulting frame
            cv2.imshow('MediaPipe Pose', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cap_user.release()
        cv2.destroyAllWindows()
        
    def display_instructions(self, frame, time):
        # Determine which subtitle to display

        for start_time, end_time, text in self.instruction.instructions:
            if start_time <= time < end_time:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7  # Adjust this value to make the text smaller or larger

                # Choose the position for the text (bottom of the frame)
                text_size, _ = cv2.getTextSize(text, font, scale, 2)
                text_width, text_height = text_size
                x = (frame.shape[1] - text_width) // 2
                y = frame.shape[0] - 50

                # Draw the outline
                outline_color = (0, 0, 0)  # black
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(frame, text, (x + dx, y + dy), font, scale, outline_color, 2, cv2.LINE_AA)

                # Draw the text
                text_color = (255, 255, 255)  # white
                cv2.putText(frame, text, (x, y), font, scale, text_color, 2, cv2.LINE_AA)

                break
       

        return frame
    

 