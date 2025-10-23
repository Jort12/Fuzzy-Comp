"""
For logging game data to CSV files
data structure:






"""
import csv, os
from pathlib import Path



# List of feature names
FEATURES = [
    "dist",
    "ttc",
    "heading_err",
    "approach_speed",
    "ammo",
    "mines",
    "threat_density",
    "threat_angle",
]

# List of action names
TARGET = ["thrust", "turn_rate"]#target action names, keep these separate from fire and mine

class Logger:
    
    def __init__(self, filepath, features, targets):
        self.filepath = filepath
        self.fieldnames = features + targets#combine features and target names
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file_exists = os.path.exists(filepath)#check if file exists

        self.file = open(self.filepath, mode='a', newline='') #append mode so it doenst overwritge old data
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames) #create writer object
        if not file_exists or os.path.getsize(filepath) == 0:#write header if file is new or empty
            self.writer.writeheader()
            
    def log(self, ctx, actions):#row_dict: Dict[str, Any]
        row = {**ctx} #copy context data
        for name, value in zip(self.fieldnames[len(ctx):], actions):#add action data
            row[name] = value
        self.writer.writerow(row)#write row to file
        self.file.flush()#ensure data is written to file
        
    def close(self):
        self.file.close()