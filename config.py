import torch
import multiprocessing

CONFIG = {
    'batch_size': 4,
    'num_epochs': 50,
    'lstm_hidden_dim': 256,
    'lstm_dropout': 0.0,
    'num_classes': 10,
    # 'rambo_mount_point': '/home/mpeven1/rambo', # GPU Server
    'rambo_mount_point': '/Users/mpeven/Rambo', # Mac
    'cuda_available': torch.cuda.is_available(),
    'cpu_count': multiprocessing.cpu_count(),
}
