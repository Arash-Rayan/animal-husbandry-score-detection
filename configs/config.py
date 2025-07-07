import argparse 
parser = argparse.ArgumentParser() 

import torch 

default_args = {
    'learning_rate': 1e-3, 
    'epochs': 1,
    'batch_size': 32, 
    'root': '/Users/alchemist/Desktop/cow_blip/frames',
    'vid_dir': '/Users/alchemist/Desktop/cow_blip/data',
    'frame_dir':'/Users/alchemist/Desktop/cow_blip/frames',
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu'
}

parser.add_argument('--learning_rate', type=int ,default= default_args['learning_rate'])
parser.add_argument('--epochs', type=int, default = default_args['epochs'])
parser.add_argument('--batch_size', type=int, default=default_args['batch_size'])
parser.add_argument('--root' , type=str, default=default_args['root'])
parser.add_argument('--vid_dir' , type=str, default=default_args['vid_dir'])
parser.add_argument('--frame_dir' , type=str, default=default_args['frame_dir'])
parser.add_argument('--device', type=str , default=default_args['device'])

args = parser.parse_args()
