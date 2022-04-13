import json
import torch
import torch.utils.data as Data

from os.path import (
    join,
    exists,
)

from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import pad
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from utils import (
    saveToPKL,
    loadFromPKL,
)

class BasicDataset(Data.Dataset):
    def __init__(self, name, split, config):
        super().__init__()
        self.name = name
        self.split = split
        # A backup for the config
        self.config = config
        
        self.path = config.data_path
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
    def __init__(self, name, split, config):
        super().__init__(name, split, config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        if self.build_from_strach:
            self.load_raw()
            self.save_cache()
            self.afterload()
        else:
            self.load_cache()
            self.afterload()
    
    def load_raw(self):
        inputFile = open(join(self.path, self.split + ".jsonl"), "r")
        data = []
        for line in inputFile:
            data_i = json.loads(line)
            # List[str]
            src_sents = data_i["inputs"]
            tgt_sents = data_i["outputs"]

            src_idx = self.tokenizer.encode(" ".join(src_sents), return_tensors="pt", truncation=True)
            tgt_idx = self.tokenizer.encode(" ".join(tgt_sents), return_tensors="pt", truncation=True)

            if src_idx.size(-1) < 3:
                print("Error: Input Empty")
                continue

            if tgt_idx.size(-1) < 3:
                print("Error: Output Empty")
                continue

            data.append([src_idx, tgt_idx])
        inputFile.close()
        self.n_data = len(data)
        self.Data = data


def data2batch(data):
    """
        data: List[List]
    """
    assert len(data) > 0
    n_field = len(data[0])
    batch_fields = []
    for i in range(n_field):
        field_i = [inst[i] for inst in data]
        batch_fields.append(field_i)
    return batch_fields


def prepare_data(batch, config):
    batch_src, batch_tgt = batch
    l_src = max([inst.size(-1) for inst in batch_src])
    batch_src_pt = torch.stack([pad(inst, (0, l_src - inst.size(-1)), value=torch.LongTensor(config.pad)) for inst in batch_src])
    l_tgt = max([inst.size(-1) for inst in batch_tgt])
    batch_tgt_pt = torch.stack([pad(inst, (0, l_tgt - inst.size(-1)), value=torch.LongTensor(config.pad)) for inst in batch_src])
    return batch_src_pt, batch_tgt_pt[:, :-1], batch_tgt_pt[:, 1:]


class Collator(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, data):
        batch = data2batch(data)
        return prepare_data(batch, self.config)


class CNNDailyMail_Module(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.collator = Collator(config)

        self.trainDataset = CNNDailymail(
            name="train",
            split="train",
            config=config,
        )

        self.validDataset = CNNDailymail(
            name="valid",
            split="valid",
            config=config,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainDataset,
            batch_size=self.config.batch_size_per_gpu,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )

    def valid_dataloader(self):
        return DataLoader(
            dataset=self.validDataset,
            batch_size=self.config.batch_size_per_gpu,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return super().test_dataloader()