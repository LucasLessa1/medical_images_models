import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


def plot_image_pred_true(model, test_dataset, device, inverted_labels_dict, plot_images=False, num_images_to_plot=10):
    # Initialize variables to store predictions and ground truth
    y_pred = []
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    images_plotted = 0  # Counter for the number of images plotted

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.round(torch.sigmoid(outputs))#.cpu().numpy()
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()  # Get the class index with the highest probability
            
            y_pred.extend(predictions)

            # Plot images with true and predicted labels
            if plot_images and images_plotted < num_images_to_plot:
                plot_image(images, labels, predictions, inverted_labels_dict)
                images_plotted += 1

            # Break the loop if the desired number of images is plotted
            if images_plotted >= num_images_to_plot:
                break


def plot_image(images, true_label, predicted_label, inverted_labels_dict):
    if len(images.shape) == 4 and images.shape[1] == 3:
        # Convert CUDA tensor to numpy array and rearrange dimensions for RGB image
        images = images.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
    else:
        # Convert CUDA tensor to numpy array and squeeze if necessary
        images = images.cpu().detach().numpy().squeeze()

    if len(images.shape) == 2:
        plt.imshow(images, cmap='gray')
    else:
        plt.imshow(images)
    
    # Format the labels
    true_label = true_label.item() if isinstance(true_label, torch.Tensor) else true_label
    predicted_label = predicted_label.item() if isinstance(predicted_label, torch.Tensor) else predicted_label
    
    plt.title(f'True label: {inverted_labels_dict[int(true_label)]}; Predicted: {inverted_labels_dict[int(predicted_label)]}')
    plt.show()
    
    
    
def plot_metrics(hist, path) :
    # plot training curves
    epochs = range(1, len(hist['train_loss']) + 1)

    train_loss = [t.cpu().detach().numpy() for t in hist['train_loss']]
    val_loss = [t.cpu().detach().numpy() for t in hist['val_loss']]


    fig, ax = plt.subplots(1,2, figsize=(20,6))
    ax[0].plot(epochs, train_loss, 'r-', label='Train')
    ax[0].plot(epochs, val_loss, 'b-', label='Evaluation')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    plt.savefig(f'{path}_Loss.png')  # This saves the plot as a PNG image

    ax[1].plot(epochs, hist['train_acc'], 'r-', label='Train')
    ax[1].plot(epochs, hist['val_acc'], 'b-', label='Evaluation')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Acc')
    ax[1].legend()
    plt.savefig(f'{path}_Accuracy.png')  # This saves the plot as a PNG image
    

    plt.show()
    