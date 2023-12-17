import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import pandas as pd

def train_model(model, train_dataset, val_dataset, test_dataset, device, path_save_modelweights, weights, 
                lr=0.0001, epochs=30, batch_size=32, l2=0.00001, gamma=0.5,
                patience=7):
    model = model.to(device)

    # Construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # History
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Set up loss function and optimizer
    if len(weights) == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        
        weight = torch.tensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)  
        
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)

    # Initialize variables to track the best validation loss and corresponding model state
    best_val_loss = float('inf')
    best_model_state = None

    # Training Loop
    print("Training Start:")
    for epoch in range(epochs):
        model.train()

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).long()   # Convert labels to Long data type

            outputs = model(images).float()  # Make sure the output is of type float
            pred = outputs.argmax(dim=1)  ####################################################
            # print(f'Pred\n{pred}')
            # print(f'labels\n{labels}')
            # print(f'outputs\n{outputs}')
            cur_train_loss = criterion(outputs, labels)
            # pred = torch.nn.functional.one_hot(pred, num_classes=num_classes).cpu().numpy()

            cur_train_acc = (pred == labels).sum().item() / batch_size

            cur_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += cur_train_loss 
            train_acc += cur_train_acc
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()   # Convert labels to Long data type

                outputs = model(images).float()  # Make sure the output is of type float

                cur_valid_loss = criterion(outputs, labels)
                val_loss += cur_valid_loss

                pred = outputs.argmax(dim=1)  ####################################################
                # pred = torch.nn.functional.one_hot(pred, num_classes=num_classes).cpu().numpy()
                val_acc += (pred == labels).sum().item() / batch_size

        scheduler.step()

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)

        print(f"Epoch:{epoch + 1} / {epochs}, lr: {optimizer.param_groups[0]['lr']:.5f} train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")
    
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
        # Update the best model if validation loss is the lowest so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_acc = 0
    print(f'The best val loss is {best_val_loss}.\n\n')
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()   # Convert labels to Long data type

            outputs = model(images).float()  # Make sure the output is of type float
            predictions = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions)
            pred = outputs.argmax(dim=1)  ####################################################
            # pred = torch.nn.functional.one_hot(pred, num_classes=num_classes).cpu().numpy()
            test_acc += (pred == labels).sum().item()
    pred_cpu = pred.cpu()  # Movendo o tensor para a CPU

    # Agora você pode convertê-lo para um array NumPy
    y_true = np.array(y_true)
    y_pred = np.array(pred_cpu)
    # y_pred = np.array(pred)
    
        # Calculate evaluation metrics
    accuracy = (test_acc / len(test_loader))
    tn, fp, fn, tp = multilabel_confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    specificity = tn / (tn + fp)
    sensitivity = tp / (fn + tp)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'Best Value Loss': best_val_loss,
        'false negativo': fn
    }
    
    dataset = pd.DataFrame(metrics.items(), columns=['Metric', 'Value']).to_csv(f'metrics.csv')
    print(f'Test Accuracy:  {accuracy}\n')
    print(f'Metrics:  {metrics}\n')

    return history, model
