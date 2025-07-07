import os 
import logging
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('fire_logger')
import torch

class Checkpoint():
    def __init__(self , device = 'cpu'):
        self.best_acc = 0
        self.device = device
        self.folder = 'checkpoint'
        os.makedirs(self.folder, exist_ok=True)

    def save(self, net, optimizer, train_losses, val_losses, acc, filename, epoch):
        if acc > self.best_acc:
            self.best_acc = acc
            logger.info('Saving checkpoint...')

            state = {
                'model': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'train_loss_list': train_losses,
                'val_loss_list': val_losses
            }

            save_path = os.path.join(self.folder, filename + '.pth')
            torch.save(state, save_path)
            
    def load(self,  root:str , name:str) :
        path = os.path.join(root, name)
        model_dict = torch.load(path , map_location=self.device)
        return model_dict

