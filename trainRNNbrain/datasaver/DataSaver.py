import json
import os
import pickle
from datetime import date
from omegaconf import OmegaConf

class DataSaver():
    '''
    Class which encapsulates creating data folders and saving information there afterward
    '''

    def __init__(self, data_folder, dj_integration=False):
        # create data folder if doesn't exist
        os.umask(0)
        os.makedirs(data_folder, exist_ok=True, mode=0o777)
        self.data_folder = data_folder
        self.date_tag = ''.join((list(str(date.today()).split("-"))[::-1]))
        self.dj_integration = dj_integration

    def save_data(self, data, file_name):
        '''save data as a pickle or json file, depending on the name'''
        if 'pkl' in file_name:
            pickle.dump(data, open(os.path.join(self.data_folder, file_name), "wb+"))
        elif 'json' in file_name:
            json_obj = json.dumps(data, indent=4)
            outfile = open(os.path.join(self.data_folder, file_name), mode="w")
            outfile.write(json_obj)
        elif 'yaml' in file_name:
            outfile = open(os.path.join(self.data_folder, file_name), mode="w")
            OmegaConf.save(data, outfile)
        return None

    def save_figure(self, figure, file_name):
        '''saving an image as a png'''
        figure.savefig(os.path.join(self.data_folder, file_name), dpi=300, format='png')
        return None

    def save_animation(self, ani, file_name):
        '''saving an animation as a mp4'''
        ani.save(os.path.join(self.data_folder, file_name), writer='ffmpeg', fps=30, dpi=300)
        return None

