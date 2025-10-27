"""
For logging game data to CSV files.

Data structure:
Each row in the CSV file contains feature values followed by target action values.
Features: dist, ttc, heading_err, approach_speed, ammo, mines, threat_density, threat_angle
Targets: thrust, turn_rate

Example row:
{
    "dist": 100.0,
    "ttc": 5.0,
    "heading_err": 0.2,
    "approach_speed": -1.5,
    "ammo": 3,
    "mines": 1,
    "threat_density": 0.7,
    "threat_angle": 45.0,
    "thrust": 0.8,
    "turn_rate": -0.1
}
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
        self.writer.writerow(row)#write row to file (row ex: {'dist': 100.0, 'ttc': 5.0, ..., 'thrust': 0.8, 'turn_rate': -0.1})
        self.file.flush()#ensure data is written to file, flush is used to clear buffer 
        
    def close(self):
        self.file.close()