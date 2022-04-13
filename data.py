import torch
import torch.utils.data as Data

from os.path import (
    join,
    exists,
)
from .utils import (
    saveToPKL,
    loadFromPKL,
)

class BasicDataset(Data.Dataset):
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        # A backup for the config
        self.config = config
        
        self.path = config.path
        self.build_from_strach = config.build_from_strach

        self.n_data = 0
        self.Data = []

    def load_raw(self):
        pass
    
    def save_cache(self):
        if self.build_from_strach:
            forSave = {
                "n_data": self.n_data,
                "Data": self.Data
            }
            saveToPKL(join(self.path, self.name+".cache"), forSave)

    def load_cache(self):
        if not self.build_from_strach:
            if not exists(join(self.path, self.name + ".cache")):
                return False
            forLoad = loadFromPKL(join(self.path, self.name + ".cache"))
            self.n_data = forLoad["n_data"]
            self.Data = forLoad["Data"]
            return True
        return False

    def afterload(self):
        pass
    
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        return self.Data[idx]


class CNNDailymail(BasicDataset):
    def __init__(self, name, config, tokenizer):
        super().__init__(name, config)
        self.tokenizer = tokenizer
        if self.build:
            self.load_raw()
            self.save_cache()
            self.afterload()
        else:
            self.load_cache()
            self.afterload()
    
    def load_raw(self):
        inputFile = 