from typing import Optional
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegressionCV
import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import snntorch as snn
import torch.nn.functional as F
from snntorch import surrogate
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn import tree
from sklearn.metrics import accuracy_score


class MyDataset(Dataset):
    def __init__(self, data, classes):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, i):
        return self.data[i], self.classes[i]

    def __len__(self):
        return len(self.data)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_path: str, train_classes_path: str, test_path: str, test_classes_path: str,
                 batch_size: int = 1):
        super().__init__()
        self.train_path = train_path
        self.train_classes_path = train_classes_path
        self.test_path = test_path
        self.test_classes_path = test_classes_path

        self.train_data = None
        self.test_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_classes = None
        self.test_classes = None
        self.sep = ';'
        self.num_classes = None
        self.batch_size = batch_size
        self.vector_len = None

        self.prepare_data()

    @staticmethod
    def pad(text_tensor, total):
        n = total - len(text_tensor)
        return F.pad(text_tensor, (0, n))

    def prepare_data(self):
        self.download_dataset()
        self.num_classes = len(set(self.train_classes))

    def download_dataset(self):
        with open(self.train_path) as f:
            self.train_data = [[float(_) for _ in line.split(self.sep)] for line in f]
        with open(self.test_path) as f:
            self.test_data = [[float(_) for _ in line.split(self.sep)] for line in f]
        with open(self.train_classes_path) as f:
            self.train_classes = [int(line) - 1 for line in f]
        with open(self.test_classes_path) as f:
            self.test_classes = [int(line) - 1 for line in f]

    def setup(self, stage: Optional[str] = None):
        self.vector_len = self.count_vector_len()
        self.train_data = [i + [0] * (self.vector_len - len(i)) for i in self.train_data]
        self.test_data = [i + [0] * (self.vector_len - len(i)) for i in
                          list(map(lambda vector: vector[:self.vector_len], self.test_data))]
        self.train_dataset = MyDataset(data=self.train_data, classes=self.train_classes)
        self.test_dataset = MyDataset(data=self.test_data, classes=self.test_classes)

    def count_vector_len(self):
        max_len = 0
        for vector in self.train_data:
            max_len = max(max_len, len(vector))
        return max_len

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    @property
    def num_inputs(self):
        if not self.vector_len:
            self.vector_len = self.count_vector_len()
        return self.vector_len

    @property
    def num_outputs(self):
        return self.num_classes


class SpikeText(pl.LightningModule):
    def __init__(self, num_inputs, num_hidden, beta, num_outputs, learning_rate):
        super().__init__()

        # Initialize layers
        self.spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        # Initialize hidden states at t=0
        self.mem1 = self.lif1.init_leaky()
        self.lr = learning_rate
        # self.loss_fct = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        self.loss_fct = F.cross_entropy
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x, mem1):
        cur1 = self.fc1(x)
        spk1, mem1_after = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        return cur2, mem1_after

    def training_step(self, batch, batch_idx, mem1=None):
        if not mem1:
            mem1 = self.mem1
        x, y = batch
        cur, mem1_after = self.forward(x, mem1)
        loss = self.loss_fct(cur, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_accuracy(cur, y)
        self.log("train_acc", self.train_accuracy.compute(), prog_bar=True)
        return {'loss': loss, 'mem1': mem1_after}

    def test_step(self, batch, batch_idx, mem1=None):
        if not mem1:
            mem1 = self.mem1
        x, y = batch
        cur, mem1_after = self.forward(x, mem1)
        loss = self.loss_fct(cur, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_accuracy(cur, y)
        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True)
        return {'loss': loss, 'mem1': mem1_after}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


ib_data_module = MyDataModule(
    train_path='data/DatasetsPM/TitleMeshIB/TrainTitleIB.csv',
    test_path='data/DatasetsPM/TitleMeshIB/TestTitleIB.csv',
    train_classes_path='data/DatasetsPM/TitleMeshIB/TrainTitleIBClass.csv',
    test_classes_path='data/DatasetsPM/TitleMeshIB/TestTitleIBClass.csv'
)
b_data_module = MyDataModule(
    train_path='data/DatasetsPM/TitleMeshB/TrainTitleB.csv',
    test_path='data/DatasetsPM/TitleMeshB/TestTitleB.csv',
    train_classes_path='data/DatasetsPM/TitleMeshB/TrainTitleBClass.csv',
    test_classes_path='data/DatasetsPM/TitleMeshB/TestTitleBClass.csv'
)
ng_data_module = MyDataModule(
    train_path='data/DatasetsPM/20NewsgroupShort/TrainShortBydate.csv',
    test_path='data/DatasetsPM/20NewsgroupShort/TestShortBydate.csv',
    train_classes_path='data/DatasetsPM/20NewsgroupShort/TrainShortBydateClass.csv',
    test_classes_path='data/DatasetsPM/20NewsgroupShort/TestShortBydateClass.csv'
)

model_ng = SpikeText(num_inputs=ng_data_module.num_inputs, num_hidden=1000, beta=0.92,
                     num_outputs=ng_data_module.num_outputs, learning_rate=1e-4)
model_ib = SpikeText(num_inputs=ib_data_module.num_inputs, num_hidden=1000, beta=0.92,
                     num_outputs=ib_data_module.num_outputs, learning_rate=1e-4)
model_b = SpikeText(num_inputs=b_data_module.num_inputs, num_hidden=1000, beta=0.92,
                    num_outputs=b_data_module.num_outputs, learning_rate=1e-4)

checkpoint_callback = ModelCheckpoint(
    monitor="train_acc",
    dirpath='lightning_logs',
    filename="{epoch:02d}-{train_acc:.2f}",
)

ib_trainer = pl.Trainer(default_root_dir='lightning_logs',
                        max_epochs=1, gpus=torch.cuda.device_count(),
                        callbacks=[checkpoint_callback])
b_trainer = pl.Trainer(default_root_dir='lightning_logs',
                       max_epochs=1, gpus=torch.cuda.device_count(),
                       callbacks=[checkpoint_callback])
ng_trainer = pl.Trainer(default_root_dir='lightning_logs',
                        max_epochs=1, gpus=torch.cuda.device_count(),
                        callbacks=[checkpoint_callback])

training_sets = [(model_ng, ng_data_module, ng_trainer), (model_b, b_data_module, b_trainer),
                 (model_ib, ib_data_module, ib_trainer)]
for model, data_module, trainer in training_sets:
    trainer.fit(model, datamodule=data_module)

for model, data_module, trainer in training_sets:
    trainer.test(model, datamodule=data_module)


class Perceptron(pl.LightningModule):
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate):
        super().__init__()
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lr = learning_rate
        self.loss_fct = F.cross_entropy
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        cur1 = self.fc1(x)
        cur2 = self.fc2(cur1)
        return cur2

    def training_step(self, batch, batch_idx):
        x, y = batch
        cur = self.forward(x)
        loss = self.loss_fct(cur, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_accuracy(cur, y)
        self.log("train_acc", self.train_accuracy.compute(), prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        cur = self.forward(x)
        loss = self.loss_fct(cur, y)
        self.log("test_loss", loss, prog_bar=True)
        self.test_accuracy(cur, y)
        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


ib_data_module = MyDataModule(
    train_path='data/DatasetsPM/TitleMeshIB/TrainTitleIB.csv',
    test_path='data/DatasetsPM/TitleMeshIB/TestTitleIB.csv',
    train_classes_path='data/DatasetsPM/TitleMeshIB/TrainTitleIBClass.csv',
    test_classes_path='data/DatasetsPM/TitleMeshIB/TestTitleIBClass.csv'
)
b_data_module = MyDataModule(
    train_path='data/DatasetsPM/TitleMeshB/TrainTitleB.csv',
    test_path='data/DatasetsPM/TitleMeshB/TestTitleB.csv',
    train_classes_path='data/DatasetsPM/TitleMeshB/TrainTitleBClass.csv',
    test_classes_path='data/DatasetsPM/TitleMeshB/TestTitleBClass.csv'
)
ng_data_module = MyDataModule(
    train_path='data/DatasetsPM/20NewsgroupShort/TrainShortBydate.csv',
    test_path='data/DatasetsPM/20NewsgroupShort/TestShortBydate.csv',
    train_classes_path='data/DatasetsPM/20NewsgroupShort/TrainShortBydateClass.csv',
    test_classes_path='data/DatasetsPM/20NewsgroupShort/TestShortBydateClass.csv'
)

ng_mlp_model = Perceptron(num_inputs=ng_data_module.num_inputs, num_hidden=1000,
                          num_outputs=ng_data_module.num_outputs, learning_rate=1e-4)
ib_mlp_model = Perceptron(num_inputs=ib_data_module.num_inputs, num_hidden=1000,
                          num_outputs=ib_data_module.num_outputs, learning_rate=1e-4)
b_mlp_model = Perceptron(num_inputs=b_data_module.num_inputs, num_hidden=1000,
                         num_outputs=b_data_module.num_outputs, learning_rate=1e-4)

checkpoint_callback = ModelCheckpoint(
    monitor="train_acc",
    dirpath='lightning_logs',
    filename="{epoch:02d}-{train_acc:.2f}",
)
mlp_trainer_ib = pl.Trainer(default_root_dir='lightning_logs',
                            max_epochs=5, gpus=torch.cuda.device_count(),
                            callbacks=[checkpoint_callback])
mlp_trainer_b = pl.Trainer(default_root_dir='lightning_logs',
                           max_epochs=5, gpus=torch.cuda.device_count(),
                           callbacks=[checkpoint_callback])
mlp_trainer_ng = pl.Trainer(default_root_dir='lightning_logs',
                            max_epochs=5, gpus=torch.cuda.device_count(),
                            callbacks=[checkpoint_callback])

mlp_training_sets = [(ng_mlp_model, ng_data_module, mlp_trainer_ng),
                     (ib_mlp_model, ib_data_module, mlp_trainer_ib),
                     (b_mlp_model, b_data_module, mlp_trainer_b)]

for mlp_model, mlp_data_module, mlp_trainer in mlp_training_sets:
    mlp_trainer.fit(mlp_model, datamodule=mlp_data_module)
    mlp_trainer.test(mlp_model, datamodule=mlp_data_module)

for data_module in [ng_data_module, ib_data_module, b_data_module]:
    X = data_module.train_data
    y = data_module.train_classes
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    pred = clf.predict(data_module.test_data)
    print("RESULT:", accuracy_score(data_module.test_classes, pred))

# for data_module in [ng_data_module, ib_data_module, b_data_module]:
data_module = ng_data_module
X = data_module.train_data
y = data_module.train_classes
lrcv = LogisticRegressionCV()
lrcv.fit(X, y)
lrcv_pred = lrcv.predict(data_module.test_data)
print("RESULT:", accuracy_score(data_module.test_classes, lrcv_pred))
