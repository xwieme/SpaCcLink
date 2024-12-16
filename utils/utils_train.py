import pandas as pd
import os
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def switchID(data, mapping):
    data.iloc[1] = data.iloc[1].map(mapping)
    data.iloc[2] = data.iloc[2].map(mapping)
    return data

def load_data(input_dir, mode = "train"):
    
    express_matrix_path = os.path.join(input_dir, "express_matrix.csv")
    pathways_path = os.path.join(input_dir, "pathways.csv")

    express_matrix = pd.read_csv(express_matrix_path, index_col=0)

    pathways = pd.read_csv(pathways_path, index_col=0)
    
    if mode == "train":
        test_data_path = os.path.join(input_dir, "test_data.csv")
        test_data = pd.read_csv(test_data_path)
        
        train_data_path = os.path.join(input_dir, "train_data.csv")
        train_data = pd.read_csv(train_data_path)
        
        return train_data, test_data, express_matrix, pathways
    elif mode == "test":
        test_data_path = os.path.join(input_dir, "test_data.csv")
        test_data = pd.read_csv(test_data_path)

        return test_data, express_matrix, pathways
    else:
        rec_tg_path = os.path.join(input_dir, "rec_tg.csv")
        rec_tg = pd.read_csv(rec_tg_path, index_col=0)
        return rec_tg, express_matrix, pathways


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles = 0.5, last_epoch = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

        
        