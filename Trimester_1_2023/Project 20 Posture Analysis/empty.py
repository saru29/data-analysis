def perform_forward_fold(self, landmarks, frame, duration=30):
    is_complete = False
    # Code for performing forward fold stretch
    # ...
    
    # Check if the stretch is in proper form
    is_form_valid = self.pose_analyzer.check_forward_fold_form(landmarks)
    
    if not is_form_valid:
        # Display tips to improve form
        cv2.putText(frame, "Improve your form: ...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Pause the timer
        pause_start_time = time.time()
        while not is_form_valid:
            ret, frame = self.capture_frame()
            if not ret:
                break
            landmarks = self.process_frame(frame)
            is_form_valid = self.pose_analyzer.check_forward_fold_form(landmarks)
            
            # Display frame and tips
            cv2.imshow('Post Workout Stretches', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        # Adjust the start time of the timer
        self.start_time += time.time() - pause_start_time
    
    # Continue the timer and check for completion
    if time.time() - self.start_time >= duration:
        is_complete = True
    
    return is_complete

def perform_shoulder_chest_stretch(self, landmarks, frame, duration=60, side=None):
    is_complete = False
    # Code for performing shoulder chest stretch
    # ...
    
    if side == 'right':
        # Check if the stretch is in proper form for the right side
        is_form_valid = self.pose_analyzer.check_shoulder_chest_stretch_form(landmarks, side='right')
        if not is_form_valid:
            # Display tips to improve form for the right side
            cv2.putText(frame, "Improve your form (Right side): ...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Pause the timer
            pause_start_time = time.time()
            while not is_form_valid:
                ret, frame = self.capture_frame()
                if not ret:
                    break
                landmarks = self.process_frame(frame)
                is_form_valid = self.pose_analyzer.check_shoulder_chest_stretch_form(landmarks, side='right')
                
                # Display frame and tips
                cv2.imshow('Post Workout Stretches', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            # Adjust the start time of the timer
            self.start_time += time.time() - pause_start_time
    elif side == 'left':
        # Check if the stretch is in proper form for the left side
        is_form_valid = self.pose_analyzer.check_shoulder_chest_stretch_form(landmarks, side='left')
        if not is_form_valid:
            # Display tips to improve form for the left side
            cv2.putText(frame, "Improve your form (Left side): ...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Pause the timer
            pause_start_time = time.time()
            while not is_form_valid:
                ret, frame = self.capture_frame()
                if not ret:
                    break
                landmarks = self.process_frame(frame)
                is_form_valid = self.pose_analyzer.check_shoulder_chest_stretch_form(landmarks, side='left')
                
                # Display frame and tips
                cv2.imshow('Post Workout Stretches', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            # Adjust the start time of the timer
            self.start_time += time.time() - pause_start_time
    
    # Continue the timer and check for completion
    if time.time() - self.start_time >= duration:
        is_complete = True
    
    return is_complete
