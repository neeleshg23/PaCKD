import os
import csv

directory = "res"

for filename in os.listdir(directory):
    if filename.endswith(".tsv"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')
            max_f1_value = float("-inf")
            max_f1_file = ""
            for row in reader:
                f1_value = float(row["F1"])
                if f1_value > max_f1_value:
                    max_f1_value = f1_value
                    max_f1_file = filename
        print(f"Max F1 value in {filename}: {max_f1_value}")
