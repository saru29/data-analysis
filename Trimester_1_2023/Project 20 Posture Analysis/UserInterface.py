from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from Analyser import PoseAnalyzer
from Stretches import PoseComparison
from UserData import UserVisualizer


class WorkoutMode:
    def __init__(self):
        self.preworkout = False
        self.postworkout = False
        self.cycling=False

    def set_preworkout(self):
        self.preworkout = True
        self.postworkout = False
        self.cycling=False

    def set_postworkout(self):
        self.preworkout = False
        self.postworkout = True
        self.cycling=False
    
    def set_cycling(self): 
        self.preworkout = False
        self.postworkout = False
        self.cycling=True

        

workout_mode = WorkoutMode()

def switch_to_stretches():
    print("Switched to stretches mode")
    btn_stretches.setStyleSheet(selected_button_style)
    btn_cycling.setStyleSheet(button_style)
    btn_delete_data.setStyleSheet(button_style)
    btn_view_data.setStyleSheet(button_style)
    btn_cooldowns.setStyleSheet(button_style)
    workout_mode.set_preworkout()
    
    pose_comparison = PoseComparison('preworkout_stretch.mp4', threshold=20)
    try:
        pose_comparison.compare()
    except Exception as e:
        print(f"Exception occurred: {e}")
        pass


def switch_to_cycling():
    print("Switched to cycling mode")
    btn_stretches.setStyleSheet(button_style)
    btn_cycling.setStyleSheet(selected_button_style)
    btn_view_data.setStyleSheet(button_style)
    btn_delete_data.setStyleSheet(button_style)
    btn_cooldowns.setStyleSheet(button_style)
    workout_mode.set_cycling()
    pose_analyzer = PoseAnalyzer()
    dataset_images = pose_analyzer.load_dataset('posturedataset.json')
    dataset_joint_angles = pose_analyzer.calculate_dataset_angles(dataset_images)
    pose_analyzer.run(dataset_joint_angles)

def switch_to_cooldowns():
    print("Switched to cooldowns mode")
    btn_stretches.setStyleSheet(button_style)
    btn_cycling.setStyleSheet(button_style)
    btn_delete_data.setStyleSheet(button_style)
    btn_view_data.setStyleSheet(button_style)
    btn_cooldowns.setStyleSheet(selected_button_style)
    workout_mode.set_postworkout()
    pose_comparison = PoseComparison('postworkout_stretch.mp4', threshold=20)
    try:
        pose_comparison.compare()
    except Exception as e:
        print(f"Exception occurred: {e}")
        pass

 
    
def delete_data():
    print("Deleting data...")
    btn_delete_data.setStyleSheet(selected_button_style)
    btn_view_data.setStyleSheet(button_style)
    btn_stretches.setStyleSheet(button_style)
    btn_cycling.setStyleSheet(button_style)
    btn_cooldowns.setStyleSheet(button_style)   
    try:
        vis = UserVisualizer()  # User selects the file here
        vis.delete_file()
    except:
        pass
    

def view_data():
    print("Viewing data...")
    btn_view_data.setStyleSheet(selected_button_style)
    btn_delete_data.setStyleSheet(button_style)
    btn_stretches.setStyleSheet(button_style)
    btn_cycling.setStyleSheet(button_style)
    btn_cooldowns.setStyleSheet(button_style)
    try:
        vis = UserVisualizer()  # User selects the file here
        vis.plot_similarity_scores() 
        vis.plot_aerodynamics_score()
    except:
        pass
            

   


app = QApplication([])

window = QWidget()
window.setWindowTitle("Workout Modes")
window.resize(800, 600)  # Set default window size (width x height)

button_style = """
    QPushButton {
        
        
    background-color: white;
    color: #eb677d;
    border-style: outset;q
    border-width: 2px;
    border-radius: 10px;
    border-color: #e7e7e7;
    font: bold 14px;
    min-width: 10em;
    padding: 20px;
 
    }

    QPushButton:hover {
    background-color: white; 
    color: #f44336; 
    border: 2px solid #f44336;
    
    
    }
    QPushButton:pressed {
    background-color: white;
    color: black;
    border: 3px solid #e7e7e7;
    }
"""

selected_button_style = """
    QPushButton {
    background-color: #f44336;
    color: white;
    border-style: outset;
    border-width: 2px;
    border-radius: 10px;
    border-color: #555555;
    font: bold 14px;
    min-width: 10em;
    padding: 20px;
    }
"""

layout = QVBoxLayout()

btn_stretches = QPushButton('WARMUPS')
btn_stretches.setStyleSheet(button_style)
btn_stretches.clicked.connect(switch_to_stretches)
layout.addWidget(btn_stretches)

btn_cycling = QPushButton('CYCLING')
btn_cycling.setStyleSheet(button_style)
btn_cycling.clicked.connect(switch_to_cycling)
layout.addWidget(btn_cycling)

btn_cooldowns = QPushButton('COOLDOWNS')
btn_cooldowns.setStyleSheet(button_style)
btn_cooldowns.clicked.connect(switch_to_cooldowns)
layout.addWidget(btn_cooldowns)

btn_view_data = QPushButton('VIEW DATA')
btn_view_data.setStyleSheet(button_style)
btn_view_data.clicked.connect(view_data)
layout.addWidget(btn_view_data)

btn_delete_data = QPushButton('DELETE DATA')
btn_delete_data.setStyleSheet(button_style)
btn_delete_data.clicked.connect(delete_data)
layout.addWidget(btn_delete_data)

window.setLayout(layout)

window.show()

app.exec_()
