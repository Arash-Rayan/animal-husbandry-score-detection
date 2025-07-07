from configs.config import args
from torch import nn
import torch
from torchmetrics import  SymmetricMeanAbsolutePercentageError


loss_fn = nn.MSELoss()

def measure_model_performance(ypred , ytrue): 
        p_loss = SymmetricMeanAbsolutePercentageError().to(args.device)
        smape = p_loss(ypred , ytrue)
        return smape 