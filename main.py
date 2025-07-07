from configs.config import args
from utils.plot_performance import plot_performnace 
from models.slowfast import slow_fast_model, train_and_evaluate
from utils.dataset.slow_fast_dataset import data_loader 
from utils.EvalMetrics import loss_fn

slow_fast_loader = data_loader()
available_models = {
    'slowfast' : slow_fast_model
}

# train_losses, train_accuracies , train_recall, val_losses, val_accuracies , val_recall= train_and_eval_model(available_models['conv2d1d'], 'conv2d1d',loader['train_loader'], loader['test_loader'], loss_fn , args.epochs)
train_losses, train_smapes , val_losses, val_smapes = train_and_evaluate(available_models['slowfast'] , 'slow_fast' , slow_fast_loader['train_loader'], slow_fast_loader['test_loader'], loss_fn , args.epochs)

plot_performnace(len(train_losses), train_losses, val_losses ,model_name='conv2d1d' , save=True)

print(
       f'train loss is {train_losses[-1]:2f}\n',
       f'train smape is {train_smapes[-1]:.2f}\n', 
       f'val loss is {val_losses[-1]:2f}\n',
       f'val smape is {val_smapes[-1]:.2f}\n', 
       )


