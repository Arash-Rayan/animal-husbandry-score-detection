import  matplotlib.pyplot as plt
import os 

def plot_performnace(epochs:int , train_losses , val_losses , model_name:str , save:bool = False): 

    fig , axes = plt.subplots(figsize=(8, 5)) 
    axes.plot(range(1 , epochs+1), train_losses ,label='train_loss' , color='blue')
    axes.plot(range(1 ,epochs+1) , val_losses , label = 'test_loss' , color='orange')

    axes.set_xlabel('epochs')
    axes.set_ylabel('loss')
    plt.legend()
    plt.show() 

    if save : 
        root = 'models/plots'
        os.makedirs('models/plots', exist_ok=True)
        save_path = os.path.join(root , f'{model_name}.png')
        fig.savefig(save_path)