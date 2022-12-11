# MTELLEY DEAKIN
# SIT374
# DATA SCIENCE AND ANALYSIS

import csv
import os
from os import listdir
import fitparse
import pytz
import glob
import pandas as pd
import re

x = 0

allowed_fields = [
    "timestamp",
    "position_lat",
    "position_long",
    "distance",
    "enhanced_altitude",
    "altitude",
    "ascent",
    "grade",
    "calories",
    "enhanced_speed",
    "speed",
    "heart_rate",
    "temperature",
    "cadence",
    # "fractional_cadence",
    "power",
    "left_right_balance",
    "gps_accuracy",
    "descriptor",
    "product_name",
    "serial_number",
    "age",
    "gender",
    "weight",
    "FTP",
    "wperkg",
    "sessionID",
    "userID"
]
required_fields = ["timestamp"]

UTC = pytz.UTC
AEST = pytz.timezone("Australia/Melbourne")

def renameFiles(x):
    print("Renaming Started")
    #Key folder name
    folder = "Exports"
    #Seperator
    sep = ""
    #Front of file name
    leadName = "activity-file-"
    #Get Path
    path = os.path.dirname(
        __file__
    )

    print(path)
    #Get File Name
    files = os.listdir()
    file_extension ="";
    #print(files)
    for index, file in enumerate(files):
        # blocks = re.split("\W+", file)
        # file_extension = blocks[(len(blocks)-1)]
        # #print(file_extension)
        # if file_extension != "py":
        #     tempfilename = sep.join(blocks[7:9])
        #     tempfilename = leadName + tempfilename
        #     #print(tempfilename)
        #     os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(tempfilename), '.fit'])))
            x+=1
    print(x, " files renamed")
    return x

def main(x):
    print("Conversion started:")
    y=0
    files = os.listdir()
    fit_files = [file for file in files if file[-4:].lower() == ".fit"]
    for file in fit_files:
        new_filename = file[:-4] + ".csv"
        if os.path.exists(new_filename):
            # print('%s already exists. skipping.' % new_filename)
            continue
        fitfile = fitparse.FitFile(
            file, data_processor=fitparse.StandardUnitsDataProcessor()
        )
        #print("converting %s" % file)
        #print(fitfile)
        failedFiles = 0;
        try:
            write_fitfile_to_csv(fitfile, new_filename)
        except:
            print("Exception thrown!")
            failedFiles+=1
        y+=1
        print("File Converted, ", round((y/x)*100,2),"%")
    print("finished conversions, failed conversions: ", failedFiles)

def write_fitfile_to_csv(fitfile, output_file="test_output.csv"):
    messages = fitfile.messages
    # raw data in messages in one line
    data = []
    for m in messages:
        skip = False
        if not hasattr(m, "fields"):
            continue
        fields = m.fields
        # check for important data types
        mdata = {}
        for field in fields:
            # print(field) print varaibles
            if field.name in allowed_fields:
                if field.name == "timestamp":
                    mdata[field.name] = UTC.localize(field.value).astimezone(UTC) #AEST
                else:
                    mdata[field.name] = field.value
        for rf in required_fields:
            if rf not in mdata:
                skip = True
        if not skip:
            data.append(mdata)
    # write to csv
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(allowed_fields)
        #print(allowed_fields)
        for entry in data:
            line_file = []
            for k in allowed_fields:
                data_var = str(entry.get(k, ""))
                #print(entry," ", k," " ,data_var)
                line_file.append(data_var)
            #print(line_file)
            writer.writerow(line_file)
    #print("wrote %s" % output_file)

def combine_csvfiles():
    print("Combining Files")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #print(all_filenames)

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    x = renameFiles(x)
    main(x)
    combine_csvfiles()
    print("OPERATIONS COMPLETE")

##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------

## Notes:

### DOCUMENTATION RENAME FUNCT
## COMMENTS
# example filename = "2022-11-18-192712-ELEMNT BOLT 480A-445-0" + fit on end
# Wahoo format (FILE NAME)
# 0 Year = 2022
# 1 Month = 11
# 2 Day = 18
# 3 TIME? = 192712
# 4:7 Device = ELEMNT', 'BOLT', '480A',  'jpg']
# 7:9 ID = '445', '0',
# 9 file type
# We want to format the filename to allow for wildcard (*) ops in google cloud re query capablitites
# activity-file-*
# * = index+ID

## Join Logic
# for f in os.listdir(path):
#     # View File Name
#     print(f)
#     # Split File Name
#     blocks = re.split("\W+", f)
#     tempfilename = sep.join(blocks[7:9])
#     tempfilename = leadName + tempfilename
#     # Check split and join (Use to rename file name)
#     print(tempfilename)
#     # Reset
#     tempfilename = ""
#     blocks = ""

### RESOURCES:
# https://support.wahoofitness.com/hc/en-us/articles/115000127910-Connecting-ELEMNT-BOLT-ROAM-to-Desktop-or-Laptop-Computers
# https://www.android.com/filetransfer/
# https://www.youtube.com/watch?v=MwN-NrkjSUY&ab_channel=franchyze923
# https://www.java.com/en/download/
# https://developer.garmin.com/fit/overview/
# https://developer.garmin.com/fit/download/
# https://www.youtube.com/watch?v=k8vrTdcHtt0&ab_channel=RDGames%26Tech
# https://github.com/rdchip/FIT-to-CSV-converter-for-windows
# https://www.python.org/downloads/release/python-3110/
# https://www.jetbrains.com/edu-products/download/#section=pycharm-edu
# https://pypi.org/project/fitparse/#description
# https://github.com/dtcooper/python-fitparse
# https://github.com/ekapope/Combine-CSV-files-in-the-folder/blob/master/Combine_CSVs.py

