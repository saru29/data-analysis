# Oxygen Uptake - Predicting VO2 using Heart Rate & Time

## Guide

Click Me !!!

[![IMAGE ALT TEXT](http://img.youtube.com/vi/DHPCkAqcQ9A/0.jpg)](http://www.youtube.com/watch?v=DHPCkAqcQ9A "Oxygen Consumption Guide")

## Workflow

The following project accepts several different types of data sources.

Regardless of data source, a pretrained model is applied to determine/extract pose landmark coordinates.

Specific landmarks are then used to determine certains types of posture during cycling.

![alt text](posture_analysis_workflow.png)

## Results

Normal Position

[![IMAGE ALT TEXT](http://img.youtube.com/vi/bwq7a58RRRQ/0.jpg)](http://www.youtube.com/watch?v=bwq7a58RRRQ "Normal Pose")

Aero Position

[![IMAGE ALT TEXT](http://img.youtube.com/vi/o7ViRmn7PLI/0.jpg)](http://www.youtube.com/watch?v=o7ViRmn7PLI "Aero Pose")

## What's already been tested:
  - Mediapipe
    - Model Complexity (0,1 & 2)

## Orignal Data Sources

Universtiy of Costa Rica Dataset:
  - Article : https://revistas.ucr.ac.cr/index.php/pem/article/view/41360 
  - Data Source : https://data.mendeley.com/datasets/vmwrtj29kr/1 

Kaggle Dataset:
  - Article & Data: https://www.kaggle.com/datasets/andreazignoli/cycling-vo2 

University of Malaga Dataset:
  - Article: https://www.tandfonline.com/doi/abs/10.1080/15438627.2021.1954513?journalCode=gspm20 
  - Data: https://physionet.org/content/treadmill-exercise-cardioresp/1.0.1/ 
