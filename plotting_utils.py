import re, os

import pandas as pd

def get_png_file_info(png_files):

    png_files_info=[]
    for png_file in png_files:
        d={}
        d['filename']=png_file
        (d['model_1_name'],
        d['model_1_target'],
        d['model_2_name'],
        d['model_2_target'],
        d['seed'])=re.findall(r'([^-]+)-(.+)_vs_([^-]+)-(.+)_seed(\d+).png',os.path.basename(png_file))[0]
        png_files_info.append(d)
    return pd.DataFrame(png_files_info)
