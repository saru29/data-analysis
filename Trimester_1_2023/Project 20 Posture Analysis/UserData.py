import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class UserVisualizer:

    def __init__(self):
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing

        self.data_file = askopenfilename()  # show an "Open" dialog box and return the path to the selected file        
        self.data = pd.read_json(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='%Y%m%d%H%M%S')


    # (rest of the methods are the same)

    def delete_file(self):
        import os
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
            print(f"File {self.data_file} has been deleted.")
        else:
            print("The file does not exist.")


    def plot_similarity_scores(self):
        plt.figure(figsize=(10, 6))
        for joint in ['hip', 'knee', 'elbow', 'wrist']:
            sns.lineplot(data=self.data, x='timestamp', y=joint+'_similarity_score', label=joint)
        plt.title('Similarity Scores Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Similarity Score')
        plt.legend()
        plt.show()

    def plot_aerodynamics_score(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.data, x='timestamp', y='aerodynamics_score', label='Aerodynamics Score')
        plt.title('Aerodynamics Score Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Aerodynamics Score')
        plt.legend()
        plt.show()




