import json
import os
from datetime import date
import pickle

class DataSaver():
    '''
    Class which encapsulates creating data folders and saving information there afterwards
    '''
    def __init__(self, data_folder):
        # create data folder if doesn't exist
        os.makedirs(data_folder, exist_ok=True)
        self.data_folder = data_folder
        self.date_tag = ''.join((list(str(date.today()).split("-"))[::-1]))

    def save_data(self, data, file_name):
        '''save data as a pickle or json file, depending on the name'''
        if 'pkl' in file_name:
            pickle.dump(data, open(os.path.join(self.data_folder, file_name), "wb+"))
        elif 'json' in file_name:
            json_obj = json.dumps(data, indent=4)
            outfile = open(os.path.join(self.data_folder, file_name), mode="w")
            outfile.write(json_obj)
        return None

    def save_figure(self, figure, file_name):
        '''saving an image as a png'''
        figure.savefig(os.path.join(self.data_folder, file_name), dpi=300, format='png')
        return None
