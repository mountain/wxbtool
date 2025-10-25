from torch.utils.data import DataLoader

from wxbtool.core.decorators import ci_batch_injection
from wxbtool.paradigm.base import LightningModel


class Seq2SeqModel(LightningModel):
    def __init__(self, model, opt=None):
        super(Seq2SeqModel, self).__init__(model, opt)

    @ci_batch_injection(batch_size_ci=2)
    def train_dataloader(self):
        if self.model.dataset_train is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        batch_size = self.opt.batch_size
        num_workers = self.opt.n_cpu
        return DataLoader(
            self.model.dataset_train,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    @ci_batch_injection(batch_size_ci=2)
    def val_dataloader(self):
        if self.model.dataset_eval is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        batch_size = self.opt.batch_size
        num_workers = self.opt.n_cpu
        return DataLoader(
            self.model.dataset_eval,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    @ci_batch_injection(batch_size_ci=2)
    def test_dataloader(self):
        if self.model.dataset_test is None:
            if self.opt.data != "":
                self.model.load_dataset("train", "client", url=self.opt.data)
            else:
                self.model.load_dataset("train", "server")

        batch_size = self.opt.batch_size
        num_workers = self.opt.n_cpu
        return DataLoader(
            self.model.dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
