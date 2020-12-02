import pandas as pd
import json

import csv
import os
import sys


class DataSorter:
    def __init__(self, db_files):
        self.db_files = db_files
        self.raw_data = self.load_experiment_results()
        self.sorted_data = self.sort_data(use_labels=True)

    def load_experiment_results(self):
        all_loaded_data = {}
        for i, db_file in enumerate(self.db_files):
            with open(db_file, 'rb+') as data:
                for line in data:
                    df = pd.DataFrame(json.loads(line))
                all_loaded_data[db_file] = df
        return all_loaded_data

    def sort_data(self, use_labels=False):
        csvs = {}
        for identifier in self.raw_data.keys():
            csvs[identifier] = {}
            df = self.raw_data[identifier]
            try:
                exploded_positions = df.all_paths.explode()
                exploded_flash_steps = df.all_flash_steps.explode()
                csvs[identifier][0] = []
                for i in range(len(exploded_flash_steps)):
                    for index, flashed in enumerate(exploded_flash_steps[i]):
                        if flashed:
                            if not use_labels:
                                to_add = (exploded_positions[i][str(index)][0],
                                          exploded_positions[i][str(index)][1],
                                          index)
                            else:
                                label = i+1
                                to_add = (exploded_positions[i][str(index)][0],
                                          exploded_positions[i][str(index)][1],
                                          label,
                                          index)
                            csvs[identifier][0].append(to_add)
            except AttributeError:
                for col in df.columns:
                    for h in range(len(df[col])):
                        csvs[identifier][h] = []
                    for i, data in enumerate(df[col]):
                        exploded_positions = data['all_paths']
                        exploded_flash_steps = data['all_flash_steps']
                        for j in range(len(exploded_flash_steps)):
                            for index, flashed in enumerate(exploded_flash_steps[j]):
                                if flashed:
                                    if not use_labels:
                                        to_add = (exploded_positions[j][str(index)][0],
                                                  exploded_positions[j][str(index)][1],
                                                  index)
                                    else:
                                        label = j+1
                                        to_add = (exploded_positions[j][str(index)][0],
                                                  exploded_positions[j][str(index)][1],
                                                  label,
                                                  index)
                                    csvs[identifier][i].append(to_add)

        return csvs

    def write_csv(self, use_labels=False):
        for identifier in self.sorted_data.keys():
            for df_id, data in self.sorted_data[identifier].items():
                if not use_labels:
                    with open('{}_trial_{}_csv.csv'.format(identifier.split('.json')[0], df_id), 'w') as file:
                        writer = csv.writer(file)
                        writer.writerows(data)
                else:
                    with open('{}_trial_{}_csv_labeled.csv'.format(identifier.split('.json')[0], df_id), 'w') as file:
                        writer = csv.writer(file)
                        writer.writerows(data)


db_path = 'data_from_cluster/raw_experiment_results/11_20/'
db_files = []
if len(sys.argv) > 1:
    for f in sys.argv[1:]:
        db_files.append(db_path+f)

db = DataSorter(db_files)
db.write_csv(use_labels=True)
