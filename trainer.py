import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import pandas as pd


labels_dict = {}

class ModelTrainer:
    def __init__(self, model, device, weights,
                train_dataset, val_dataset, test_dataset, 
                lr=0.0001, epochs=30, batch_size=32, 
                l2=0.00001, gamma=0.5, patience=7):
        
        self.device = device
        self.model = model.to(self.device)
        self.weights = weights
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.gamma = gamma
        self.patience = patience
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
    
    def loader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
    
    def loss_function(self):
        self.best_val_loss = float('inf')
        if len(self.weights) == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            
            self.weight = torch.tensor(self.weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.weight)  
    def optimizer_step(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.patience, gamma=self.gamma)
        
        
        
    def train(self):



        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).long()   # Convert labels to Long data type

            outputs = self.model(images).float()  # Make sure the output is of type float
            pred = outputs.argmax(dim=1)  
            cur_train_loss = self.criterion(outputs, labels)

            cur_train_acc = (pred == labels).sum().item() / self.batch_size

            self.optimizer.zero_grad()
            cur_train_loss.backward()
            self.optimizer.step()

            self.train_loss += cur_train_loss 
            self.train_acc += cur_train_acc

    def validate(self):
        self.best_model_state = None
        self.val_loss = 0
        self.val_acc = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()   # Convert labels to Long data type

                outputs = self.model(images).float()  # Make sure the output is of type float

                cur_valid_loss = self.criterion(outputs, labels)
                self.val_loss += cur_valid_loss

                pred = outputs.argmax(dim=1)  
                self.val_acc += (pred == labels).sum().item() / self.batch_size

        self.scheduler.step()




    def test(self):
        y_true = []
        y_pred = []
        self.test_acc = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device).long()

                outputs = self.model(images).float()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # Get the class index with the highest probability

                y_true.extend(labels.cpu().numpy())
                y_pred.append(predictions)  # Append all predictions from this batch
                self.test_acc += sum((predictions == labels.cpu().numpy()).flatten())

        # Flatten the list of predictions before converting it to a NumPy array
        self.y_pred = np.concatenate(y_pred)
        self.y_true = np.array(y_true)
    def metrics(self):
        metrics = []
        conf_matrix  = multilabel_confusion_matrix(self.y_true, self.y_pred)
        accuracy = (self.test_acc / len(self.test_loader))
        tn_index = (0, 0)
        fp_index = (0, 1)
        fn_index = (1, 0)
        tp_index = (1, 1)
        for class_name, index in labels_dict.items():

            tn = conf_matrix[index][tn_index]
            fp = conf_matrix[index][fp_index]
            fn = conf_matrix[index][fn_index]
            tp = conf_matrix[index][tp_index]

            precision = precision_score(self.y_true, self.y_pred, average='weighted')
            recall = recall_score(self.y_true, self.y_pred, average='weighted')
            f1 = f1_score(self.y_true, self.y_pred, average='weighted')
            specificity = tn / (tn + fp)
            sensitivity = tp / (fn + tp)

            metrics.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Accuracy': accuracy,
                'True Negatives': tn,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'False Positives': fp,
                'False Negatives': fn,
                'True Positives': tp
            })

        # Create a DataFrame from the metrics
        self.metrics_df = pd.DataFrame(metrics)

        # Save to CSV
        self.metrics_df.to_csv(f'Model_{self.model.get_name()}__Epoch_{self.epochs}__Batch_{self.batch_size}__Accuracy_{accuracy}.csv', index=False)

    def loss_acc(self):
        
        self.train_loss = self.train_loss / len(self.train_loader)
        self.train_acc = self.train_acc / len(self.train_loader)
        self.val_loss = self.val_loss / len(self.val_loader)
        self.val_acc = self.val_acc / len(self.val_loader)
        self.history['train_loss'].append(self.train_loss)
        self.history['train_acc'].append(self.train_acc)
        self.history['val_loss'].append(self.val_loss)
        self.history['val_acc'].append(self.val_acc)

        
    def training(self):
        
        self.loader()
        self.loss_function()
        self.optimizer_step()
        print("Training Start:")
        for epoch in range(self.epochs):
            self.model.train()

            self.train_loss = 0
            self.train_acc = 0
        
            self.train()
            self.validate()
            self.loss_acc()


            print(f"Epoch:{epoch + 1} / {self.epochs}, lr: {self.optimizer.param_groups[0]['lr']:.5f} train loss:{self.train_loss:.5f}, train acc: {self.train_acc:.5f}, valid loss:{self.val_loss:.5f}, valid acc:{self.val_acc:.5f}")
                
            # Update the best model if validation loss is the lowest so far
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.best_model_state = self.model.state_dict()

            print(f'The best val loss is {self.best_val_loss}.\n')
            
            # Load the best model state
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
            
            
        self.test()
        self.metrics()
        
        return self.history, self.model, self.metrics_df
        
        
        # Example usage:
# Initialize model, datasets, and other required variables
# model = YourModel()
# train_dataset = YourTrainDataset()
# val_dataset = YourValDataset()
# test_dataset = YourTestDataset()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weights = ...

# Create an instance of ModelTrainer and call the 'training' method
# trainer = ModelTrainer(model, device, weights)
# trainer.training(train_dataset, val_dataset, test_dataset)
