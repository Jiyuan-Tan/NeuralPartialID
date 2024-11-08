from abc import ABC
import numpy as np

import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from typing import Optional

from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class ToyDataModule(LightningDataModule, ABC):
    def __init__(self, data, batch_size: int, num_workers: int):
        super().__init__()
        self.X = data
        # self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.X_train, self.X_test = None, None

    def setup(self, stage=None):
        # if self.test_size == 0:
        self.X_train, self.X_test = self.X, self.X
        # else:
        #     self.X_train, self.X_test = train_test_split(self.X, test_size=self.test_size)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.Tensor(self.X_train)), batch_size=self.batch_size,
                          num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(TensorDataset(torch.Tensor(self.X_test)), batch_size=self.batch_size,
    #                       num_workers=self.num_workers)


def init_or_resume_wandb_run(wandb_id_file_path: Path,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.

        Returns the config, if it's not None it will also update it first

        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        logger = WandbLogger(project=project_name,
                             name=run_name,
                             resume=resume_id)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        logger = WandbLogger(project=project_name, name=run_name)
        wandb_id_file_path.write_text(str(logger.experiment.id))

    return logger


def save_results(results: dict, save_path: str):
    """Save the results to a csv file"""
    df = pd.DataFrame(results)
    if Path(save_path).is_file():
        df = pd.concat([pd.read_csv(save_path), df])
    df.to_csv(save_path, index=False)


def get_results(metrics: list, monitor: str, alpha: float, coeff: float):
    distances = []
    estimands = []
    weg = []
    for mod in ['min', 'max']:
        d = f'distance_{mod}_network'
        e = f'{mod}_{monitor}'
        #f = f'{mod}_weight'
        dists = np.array([i[d] for i in metrics if d in i])
        ests = np.array([i[e] for i in metrics if e in i])
        #weight = np.array([i[f] for i in metrics if f in i])
        # distances.append(dists[-1])
        if alpha == 0:
            alpha = min(dists)
        if mod == 'min':
            min_estimand = -np.inf
            min_loss = np.inf
            #min_weight = 0
            min_dist = 0
            for i in range(len(ests)):
                if dists[i] <= coeff * alpha and min_loss > ests[i]:# + weight[i]:
                    min_loss = ests[i] # + weight[i]
                    min_estimand = ests[i]
                    #min_weight = weight[i]
                    min_dist = dists[i]
            if min_estimand == -np.inf:
                min_dist_index = np.argmin(dists)             
                min_estimand = ests[min_dist_index]
                min_dist = dists[min_dist_index]
                # min_weight = weight[min_dist_index]

            estimands.append(min_estimand)
            #weg.append(min_weight)
            distances.append(min_dist)
        else:
            max_loss = np.inf
            max_estimand = np.inf
            #max_weight = 0
            max_dist = 0
            for i in range(len(ests)):
                if dists[i] <= coeff * alpha and max_loss > -ests[i]: #+ weight[i]:
                    max_loss = -ests[i] #+ weight[i]
                    max_estimand = ests[i]
                    #max_weight = weight[i]
                    max_dist = dists[i]
            if max_estimand == np.inf:
                max_dist_index = np.argmin(dists)              
                max_estimand = ests[max_dist_index]
                max_dist = dists[max_dist_index]
                #max_weight = weight[max_dist_index]

            estimands.append(max_estimand)
            #weg.append(max_weight)
            distances.append(max_dist)

    return distances, estimands

