import os
import json

# Specify the folder path
folder_path = r'C:\Users\61493\Documents\SIT374\Posture analysis\Cycling Dataset'

# Define the target size for resizing
target_size = (512, 512)  # Adjust as needed

# Initialize data structures
image_paths = []

# Iterate through files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        # Save the image path
        image_path = os.path.join(folder_path, file_name)
        image_paths.append(image_path)

# Save the image paths to a JSON file
save_path = r'C:\Users\61493\Documents\SIT374\Posture analysis\posturedataset.json'
with open(save_path, 'w') as file:
    json.dump(image_paths, file)