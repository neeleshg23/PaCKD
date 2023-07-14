import os
import csv

directory = "res/i"

for filename in os.listdir(directory):
    if filename.endswith('res.tsv'):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')
            max_f1_value = float("-inf")
            max_f1_file = ""
            for row in reader:
                f1_value = float(row["F1"])
                p_value = float(row["P"])
                r_value = float(row['R']) 
                if f1_value > max_f1_value:
                    max_f1_value = f1_value
                    max_f1_file = filename
        print(f"Max P, R value in {filename}: {p_value} {r_value}")
