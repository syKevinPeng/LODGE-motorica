import numpy as np
import torch, sys

# from mld.data.humanml.scripts.motion_process import (process_file,
#                                                      recover_from_ric)

# from .BaseData_Module import BASEDataModule
from .FineDance_dataset import FineDance_Smpl
from .Motorica_dataset import Motorica_Smpl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .render_joints.smplfk import ax_from_6v, SMPLSkeleton


class MotoricaDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 name,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
        self.kwargs = kwargs
        self.is_mm = False
        self.smpl_fk = SMPLSkeleton()        
        # self.save_hyperparameters(logger=False)
        # self.njoints = 52       # 55
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = Motorica_Smpl(args=self.cfg, istrain=True, dataname=self.name)
            self.valset = Motorica_Smpl(args=self.cfg, istrain=False, dataname=self.name)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = Motorica_Smpl(args=self.cfg, istrain=False, dataname=self.name)
        
            
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.cfg.EVAL.BATCH_SIZE, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.cfg.TEST.BATCH_SIZE, num_workers=self.num_workers, shuffle=False, drop_last=True)
        

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            # self.test_dataset.name_list = []
            # self.test_dataset.name_list.append(self.name)
            # self.name_list = self.test_dataset.name_list
            # self.mm_list = np.random.choice(self.name_list,
            #                                 self.cfg.TEST.MM_NUM_SAMPLES,
            #                                 replace=False)
            # self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
    
if __name__ == "__main__":
    config_file = "/fs/nexus-projects/PhysicsFall/LODGE/configs/lodge/finedance_fea139.yaml"
    data_config = "/fs/nexus-projects/PhysicsFall/LODGE/configs/data/assets.yaml"
    from omegaconf import OmegaConf
    cfg_exp = OmegaConf.load(config_file)
    train_config  = cfg_exp.TRAIN
    cfg_data = OmegaConf.load(data_config)
    cfg = OmegaConf.merge(cfg_exp, cfg_data)
    print("dataset", cfg.TRAIN.DATASETS)
    trainset = FineDance_Smpl(args=cfg, istrain=True, dataname=cfg.TRAIN.DATASETS[0])

