import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


# Configuration
from config import CONFIG
HIDDEN_DIM_SIZE     = CONFIG['lstm_hidden_dim']
LSTM_DROPOUT        = CONFIG['lstm_dropout']
DATASET_NUM_CLASSES = CONFIG['num_classes']



class Model_1(nn.Module):
    ''' Spatial '''
    def __init__(self):
        super(Model_1, self).__init__()
        # Set up base image feature extractor
        self.base_model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)

        # Freeze weights
        for param in list(self.base_model.parameters())[:30]:
            param.requires_grad = False

        # LSTM Layer
        self.lstmlayer = nn.LSTM(
            input_size  = base_model_fc_size,
            hidden_size = HIDDEN_DIM_SIZE,
            dropout     = LSTM_DROPOUT
        )

        # Final layer
        self.preds = nn.Linear(HIDDEN_DIM_SIZE, DATASET_NUM_CLASSES)

    def forward(self, X):
        # Stack individual image features
        base_model_out = [self.base_model(X[:,i]) for i in range(X.size()[1])]
        x = torch.stack(base_model_out)
        x = x.view(x.size(0), x.size(1), int(np.prod(x.size()[2:]))) # Flatten

        # LSTM & final layer
        x, _ = self.lstmlayer(x)
        return self.preds(x[-1])
