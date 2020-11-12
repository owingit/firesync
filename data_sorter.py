import pandas as pd

import csv
import os
import sys


class DataSorter:
    def __init__(self, db_files):
        self.db_files = db_files
        self.raw_data = self.load_experiment_results()
        self.sorted_data = self.sort_data()

    def load_experiment_results(self):
        all_loaded_data = {}
        for i, db_file in enumerate(self.db_files):
            with open(db_file, 'rb+') as data:
                df = pd.read_json(data, orient='index')
                all_loaded_data[db_file] = df
        return all_loaded_data

    def sort_data(self):
        csvs = {}
        for identifier in self.raw_data.keys():
            csvs[identifier] = []
            df = self.raw_data[identifier]
            exploded_positions = df.all_paths.explode()
            exploded_flash_steps = df.all_flash_steps.explode()
            for i in range(len(exploded_flash_steps)):
                for index, flashed in enumerate(exploded_flash_steps[i]):
                    if flashed:
                        csvs[identifier].append((exploded_positions[i][str(index)][0], exploded_positions[i][str(index)][1], index))
        return csvs

    def write_csv(self):
        for identifier in self.sorted_data:
            with open('{}_csv.csv'.format(identifier.split('.json')[0]), 'w') as file:
                writer = csv.writer(file)
                writer.writerows(self.sorted_data[identifier])


db_path = 'data/raw_experiment_results/'
db_files = []
if len(sys.argv) > 1:
    for f in sys.argv[1:]:
        db_files.append(db_path+f)

db = DataSorter(db_files)
db.write_csv()
