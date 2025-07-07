import os 
from configs.config import args

def clean_data(data_path:str): 
    clean_score = []
    videos = os.listdir(data_path)
    for vid in videos : 
        try: 
            score = vid.split('-')[-1].replace('.mp4', '')
            clean_score.append((vid , float(score)))
        except: 
            os.remove(os.path.join(data_path, vid))
            continue
    return clean_score
