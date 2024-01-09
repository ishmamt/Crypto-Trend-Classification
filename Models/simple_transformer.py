import math
import os
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader

from evals import calculate_eval_metrics, tabulate_results


class CryptoDataset(Dataset):
    """
    Class for handling the price dataset using the pytorch wrapper.

    Attributes:
    X (torch.tensor): Features from the dataset.
    y (torch.tensor): Label from the dataset.
    """

    def __init__(self, X, y, dtype=torch.float32):
        """
        This is the constructor that transforms dataframes and series to pytorch tensors.

        Parameters:
        X (pd.Dataframe): Features from the dataset.
        y (pd.Series): Label from the dataset.
        dtype (torch.dtype): Datatype of the tensor (default is torch.float32).
        """

        super().__init__()
        self.X = torch.tensor(X.values, dtype=dtype)
        self.y = torch.tensor(y.values, dtype=dtype)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        lenght (int): Lenght of the dataset.
        """

        return len(self.y)

    def __getitem__(self, index):
        """
        Returns a single entry given the index from the dataset.

        Parameters:
        index (int): Index of the item to be fetched.
        
        Returns:
        entry (list): Single entry from the dataset as a list of features and target label.
        """

        return self.X[index].unsqueeze(1), self.y[index]


class CryptoDataLoader(L.LightningDataModule):
    """
    Class for handling the price dataset using the pytorch lightning dataloader.

    Attributes:
    batch_size (int): Training batch size.
    data_path (str): Path to the stored dataset.
    """

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size, test_size=0.2):
        """
        This is the constructor that creates the Data loader.

        Parameters:
        X_train (pd.DataFrame): A DataFrame containing the training data split.
        X_val (pd.DataFrame): A DataFrame containing the validation data split.
        X_test (pd.DataFrame): A DataFrame containing the testing data split.
        y_train (pd.Series): A Series containing the labels for the training data split.
        y_val (pd.Series): A Series containing the labels for the validation data split.
        y_test (pd.Series): A Series containing the labels for the testing data split.
        batch_size(int): batch size for training.
        test_size (float): The proportion of the dataset to include in the validation split (default is 0.2).
        """

        super().__init__()
        self.batch_size = batch_size
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.test_size = test_size

    def setup(self, stage=None):
        """
        This method assigns the train, val and test datasets for use in dataloaders.
        """

        self.train_dataset = CryptoDataset(self.X_train, self.y_train)
        self.val_dataset = CryptoDataset(self.X_val, self.y_val)
        self.test_dataset = CryptoDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        """
        This method returns the train dataloader.

        Returns:
        train_dataloader (DataLoader): The train dataloader.
        """

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        """
        This method returns the validation dataloader.

        Returns:
        val_dataloader (DataLoader): The validation dataloader.
        """

        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """
        This method returns the test dataloader.

        Returns:
        test_dataloader (DataLoader): The test dataloader.
        """

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class PostionalEncoding(nn.Module):
    def __init__(
        self,
        dropout: float=0.1,
        max_seq_len: int=5000,
        d_model: int=512,
        batch_first: bool=False    ):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(self.x_dim)]

        x = self.dropout(x)
        return x


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, details):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.details = details
    def forward(self, q, k, v ,e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose

        if self.details: print('in Scale Dot Product, k_t size: '+ str(k_t.size()))
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product


        if self.details: print('in Scale Dot Product, score size: '+ str(score.size()))
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        if self.details: print('in Scale Dot Product, score size after softmax : '+ str(score.size()))

        if self.details: print('in Scale Dot Product, v size: '+ str(v.size()))
        # 4. multiply with Value
        v = score @ v

        if self.details: print('in Scale Dot Product, v size after matmul: '+ str(v.size()))
        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, details):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention( details=details)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.details = details

    def forward(self, q, k, v ):
        # 1. dot product with weight matrices

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        if self.details: print('in Multi Head Attention Q,K,V: '+ str(q.size()))
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        if self.details: print('in splitted Multi Head Attention Q,K,V: '+ str(q.size()))
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v )

        if self.details: print('in Multi Head Attention, score value size: '+ str(out.size()))
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        if self.details: print('in Multi Head Attention, score value size after concat : '+ str(out.size()))
        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob,details):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, details=details)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.details = details
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x )

        if self.details: print('in encoder layer : '+ str(x.size()))
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if self.details: print('in encoder after norm layer : '+ str(x.size()))
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        if self.details: print('in encoder after ffn : '+ str(x.size()))
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob,details, device):
        super().__init__()


        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x ):
        for layer in self.layers:
            x = layer(x )
        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, seq_len, details, n_classes=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.details = details

        # Adjusted the Linear layers to handle reshaped input
        self.fc1 = nn.Linear(d_model * seq_len, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        if self.details:
            print('in classification head : ' + str(x.size()))

        x = self.norm(x)
        # Reshape the input tensor before passing through Linear layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        if self.details:
            print('in classification head after seq: ' + str(x.size()))

        return x


class Transformer(L.LightningModule):
    def __init__(self, d_model, n_head, max_len, seq_len, ffn_hidden, n_layers, drop_prob, device, details=False):
        super().__init__()
        self.details = details
        self.encoder_input_layer = nn.Linear(in_features=1, out_features=d_model)
        self.pos_emb = PostionalEncoding(max_seq_len=max_len, batch_first=False, d_model=d_model, dropout=0.1)
        self.encoder = Encoder(d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob, n_layers=n_layers, details=details, device=device)
        self.classHead = ClassificationHead(seq_len=seq_len, d_model=d_model, details=details, n_classes=2)

    def forward(self, src):
        if self.details:
            print('before input layer: ' + str(src.size()))
        src = self.encoder_input_layer(src)
        if self.details:
            print('after input layer: ' + str(src.size()))
        src = self.pos_emb(src)
        if self.details:
            print('after pos_emb: ' + str(src.size()))
        enc_src = self.encoder(src)
        cls_res = self.classHead(enc_src)
        if self.details:
            print('after cls_res: ' + str(cls_res.size()))
        return cls_res

    def cross_entropy_loss(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        loss_class = criterion(pred.squeeze(-1), target.squeeze(-1).long())
        return loss_class

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.cross_entropy_loss(outputs.squeeze(-1), labels.squeeze(-1).long())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.cross_entropy_loss(outputs.squeeze(-1), labels.squeeze(-1).long())
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(outputs)
        _, pred = torch.max(pred_probs, dim=1)
        correct = torch.sum(pred == labels).item()
        total = labels.size(0)

        y_true = labels.cpu().numpy()
        y_pred = pred.cpu().numpy()

        return {
            'val_loss': loss,
            'correct': correct,
            'total': total,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.cross_entropy_loss(outputs.squeeze(-1), labels.squeeze(-1).long())
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(outputs)
        _, pred = torch.max(pred_probs, dim=1)
        correct = torch.sum(pred == labels).item()
        total = labels.size(0)

        y_true = labels.cpu().numpy()
        y_pred = pred.cpu().numpy()

        return {
            'test_loss': loss,
            'correct': correct,
            'total': total,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

class SimpleTransformer():
    """
    Class for transformer model.

    Attributes:
    name (str): Name of the model kept for creating the results table.
    model (Transformer): The model object that will be trained and evaluated.
    """

    def __init__(self, data_dir, datasets, name="Simple Transformer", random_state=42):
        """
        This method gets called for each new instace of the class.

        Parameters:
        data_dir (str): Path to the dataset.
        datasets (list): Filenames of the dataset splits.
        name (str): Name of the model kept for creating the results table.
        random_state (int): A random seed for reproducibility (default is 42).
        """

        # Set the parameters
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_len = 6 # number of features
        self.max_len = 5000 # max time series sequence length
        self.n_head = 8 # number of attention head
        self.n_layer = 7 # number of encoder layer
        self.drop_prob = 0.1
        self.d_model = 512 # number of dimension (for positional embedding) [d_model/n_head]
        self.ffn_hidden = 128 # size of hidden layer before classification
        self.feature = 1 # for univariate time series (1d), it must be adjusted for 1.
        self.batch_size = 32
        self.max_epochs = 5

        # Set seed for everything to reproduce code
        torch.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        L.seed_everything(random_state)

        #For the dataloader
        self.X_train = pd.read_csv(os.path.join(data_dir, datasets[0]))
        self.X_val = pd.read_csv(os.path.join(data_dir, datasets[1]))
        self.X_test = pd.read_csv(os.path.join(data_dir, datasets[2]))
        self.y_train = pd.read_csv(os.path.join(data_dir, datasets[3]))
        self.y_val = pd.read_csv(os.path.join(data_dir, datasets[4]))
        self.y_test = pd.read_csv(os.path.join(data_dir, datasets[5]))

        # Initialize LightningDataModule
        self.data_module = CryptoDataLoader(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.batch_size)

        # Initialize LightningModule
        self.model = Transformer(d_model=self.d_model, n_head=self.n_head, max_len=self.max_len, seq_len=self.sequence_len, 
                                 ffn_hidden=self.ffn_hidden, n_layers=self.n_layer, drop_prob=self.drop_prob, device=self.device, details=False)
        self.model = self.model.to(torch.float32)
        self.model.to(device=self.device)

        # Initialize Trainer
        self.trainer = L.Trainer(max_epochs=self.max_epochs)

    
    def train(self, verbose=False):
        """
        Trains the model on the given training data.
        """

        # Train the model
        self.trainer.fit(self.model, self.data_module)

        if verbose:
            # Calculate and print accuracy, F1 score, recall, precision, and ROC AUC on training data
            train_predictions = []
            train_labels = []
            for batch in self.data_module.train_dataloader():
                inputs, labels = batch
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                train_predictions.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            model_performance = calculate_eval_metrics(train_predictions, train_predictions, train_labels, self.name)
            print(tabulate_results(model_performance))


    def validate(self):
        """
        Validates the trained model.
        """

        # Calculate and print accuracy, F1 score, recall, precision, and ROC AUC on validation data
        val_predictions = []
        val_labels = []
        for batch in self.data_module.val_dataloader():
            inputs, labels = batch
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, dim=1)
            val_predictions.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

        model_performance = calculate_eval_metrics(val_predictions, val_predictions, val_labels, self.name)
        print(tabulate_results(model_performance))


    def test(self):
        """
        Tests the trained model.

        Returns:
        model_performance (list): A list containing Accuracy, Precision, Recall, F1 Score and ROC AUC scores.
        """

        # Calculate and print accuracy, F1 score, recall, precision, and ROC AUC on test data
        test_predictions = []
        test_labels = []
        for batch in self.data_module.test_dataloader():
            inputs, labels = batch
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        model_performance = calculate_eval_metrics(test_predictions, test_predictions, test_labels, self.name)
        
        return model_performance