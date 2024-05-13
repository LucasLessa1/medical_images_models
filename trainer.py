import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Union, Any


class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 weights: Union[List[float]],
                 labels_dict: Dict[int, str],
                 train_dataset: Any,
                 val_dataset: Any,
                 test_dataset: Any,
                 learn_rate: float = 0.0001,
                 epochs: int = 30,
                 batch_size: int = 32,
                 l2_regul_coeff: float = 0.00001,
                 gamma: float = 0.5,
                 patience: int = 7):
        """
        Initialize the ModelTrainer.

        Args:
            model (nn.Module): The neural network model.
            device (torch.device): The device to run the training on
            (e.g., 'cuda' for GPU or 'cpu').
            weights (Union[List[float]): Weights for loss function
            (if multiple classes).
            labels_dict (Dict[int, str]): Dictionary mapping class
            indices to class labels.
            train_dataset (Any): Training dataset.
            val_dataset (Any): Validation dataset.
            test_dataset (Any): Test dataset.
            learn_rate (float, optional): Learning rate for the optimizer
            default: 0.0001).
            epochs (int, optional): Number of epochs for training
            (default: 30).
            batch_size (int, optional): Batch size for training
            (default: 32).
            l2 (float, optional): L2 regularization coefficient
            (default: 0.00001).
            gamma (float, optional): Multiplicative factor of learning
            rate decay (default: 0.5).
            patience (int, optional): Number of epochs with no improvement
            after which learning rate will be reduced (default: 7).
        """
        self.device = device
        self.model = model.to(self.device)
        self.weights = weights
        self.labels_dict = labels_dict
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = l2_regul_coeff
        self.gamma = gamma
        self.patience = patience
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0
        self.test_acc = 0
        self.optimizer = None
        self.metrics_df = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.y_pred = []
        self.y_true = []
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [],
                        'train_acc': [],
                        'val_loss': [],
                        'val_acc': []}
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

    def loader(self):
        """
        Initialize data loaders for training, validation, and testing datasets.
        """
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=1,
                                      shuffle=False)

    def loss_function(self):
        """
        Initialize loss function based on the number of classes.
        """
        self.best_val_loss = float('inf')
        if len(self.weights) == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.weights = torch.tensor(self.weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def optimizer_step(self):
        """
        Initialize optimizer and learning rate scheduler.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learn_rate,
                                          weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=self.patience,
                                                   gamma=self.gamma)

    def train(self):
        """
        Train the model.
        """
        for epoch in range(self.epochs):
            self.model.train()
            self.train_loss = 0
            self.train_acc = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images).float()
                pred = torch.argmax(outputs, dim=1)
                cur_train_loss = self.criterion(outputs, labels)

                cur_train_acc = (pred == labels).sum().item() / self.batch_size

                self.optimizer.zero_grad()
                cur_train_loss.backward()
                self.optimizer.step()

                self.train_loss += cur_train_loss
                self.train_acc += cur_train_acc

            self.train_loss = self.train_loss / len(self.train_loader)
            self.train_acc = self.train_acc / len(self.train_loader)
            self.history['train_loss'].append(self.train_loss)
            self.history['train_acc'].append(self.train_acc)

            self.validate()
            self.loss_acc()

            if self.early_stop_condition():
                print(f'Early stopping at epoch {epoch + 1}')
                break

    def validate(self):
        """
        Validate the model.
        """
        self.val_loss = 0
        self.val_acc = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images).float()
                cur_valid_loss = self.criterion(outputs, labels)
                self.val_loss += cur_valid_loss

                pred = outputs.argmax(dim=1)
                self.val_acc += (pred == labels).sum().item() / self.batch_size

        self.scheduler.step()

    def test(self):
        """
        Test the model.
        """
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                outputs = self.model(images).float()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                y_true.extend(labels.cpu().numpy())
                y_pred.append(predictions)
                self.test_acc += sum((predictions == labels.cpu().numpy())
                                     .flatten())

        self.y_pred = np.concatenate(y_pred)
        self.y_true = np.array(y_true)

    def metrics(self):
        """
        Compute evaluation metrics.
        """
        metrics = []
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        for i, class_name in zip(range(conf_matrix.shape[0]),
                                 self.labels_dict.items()):
            true_p = conf_matrix[i][i]
            false_n = np.sum(conf_matrix[i, :]) - true_p
            false_p = np.sum(conf_matrix[:, i]) - true_p
            true_n = np.sum(conf_matrix) - (true_p + false_n + false_p)

            precision = true_p / (true_p + false_p)
            recall = true_p / (true_p + false_n)
            f1_score = 2 * (precision * recall) / (precision + recall)
            specificity = true_n / (true_n + false_p)
            sensitivity = true_p / (false_n + true_p)

            metrics.append({
                'Class': class_name[1],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'Accuracy': (self.test_acc / len(self.test_loader)),
                'Best Val Loss': self.best_val_loss,
                'True Negatives': true_n,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'False Positives': false_p,
                'False Negatives': false_n,
                'True Positives': true_p
            })

        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df

    def loss_acc(self):
        """
        Compute and store training and validation losses and accuracies.
        """
        self.val_loss = self.val_loss / len(self.val_loader)
        self.val_acc = self.val_acc / len(self.val_loader)
        self.history['val_loss'].append(self.val_loss)
        self.history['val_acc'].append(self.val_acc)

    def early_stop_condition(self) -> bool:
        """
        Check if early stopping condition is met.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        if len(self.history['val_loss']) > self.patience:
            recent_losses = self.history['val_loss'][-self.patience:]
            return all(recent_losses[i] >= recent_losses[i + 1]
                       for i in range(self.patience - 1))

        return False
