import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


def plot_image_pred_true(model: torch.nn.Module,
                         test_dataset: torch.utils.data.Dataset,
                         device: torch.device,
                         inverted_labels_dict: dict,
                         plot_images: bool = False,
                         num_images_to_plot: int = 10) -> None:
    """
    Plot predicted and true labels for images from the test dataset.

    Args:
    - model (torch.nn.Module): The trained model.
    - test_dataset (torch.utils.data.Dataset): The test dataset.
    - device (torch.device): The device to perform inference on
    (e.g., "cuda" or "cpu").
    - inverted_labels_dict (dict): A dictionary mapping class
    indices to class labels.
    - plot_images (bool, optional): Whether to plot images with
    true and predicted labels (default: False).
    - num_images_to_plot (int, optional): The number of images
    to plot (default: 10).
    """
    # Initialize variables to store predictions and ground truth
    y_pred = []
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    images_plotted = 0  # Counter for the number of images plotted

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.round(torch.sigmoid(outputs))
            # Get the class index with the highest probability
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()
            y_pred.extend(predictions)

            # Plot images with true and predicted labels
            if plot_images and images_plotted < num_images_to_plot:
                plot_image(images, labels, predictions, inverted_labels_dict)
                images_plotted += 1

            # Break the loop if the desired number of images is plotted
            if images_plotted >= num_images_to_plot:
                break


def plot_image(images: torch.Tensor,
               true_label,
               predicted_label,
               inverted_labels_dict: dict) -> None:
    """
    Plot an image with true and predicted labels.

    Args:
    - images (torch.Tensor): The image tensor.
    - true_label: The true label of the image.
    - predicted_label: The predicted label of the image.
    - inverted_labels_dict (dict): A dictionary mapping class
    indices to class labels.
    """
    if len(images.shape) == 4 and images.shape[1] == 3:
        # Convert CUDA tensor to numpy array and rearrange dimensions
        images = images.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
    else:
        # Convert CUDA tensor to numpy array and squeeze if necessary
        images = images.cpu().detach().numpy().squeeze()

    if len(images.shape) == 2:
        plt.imshow(images, cmap='gray')
    else:
        plt.imshow(images)

    # Format the labels
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.item()

    if isinstance(predicted_label, torch.Tensor):
        predicted_label = predicted_label.item()

    plt.title(f'True label: {inverted_labels_dict[int(true_label)]}'
              f'; Predicted: {inverted_labels_dict[int(predicted_label)]}')
    plt.show()


def plot_metrics(hist: dict,
                 path: str) -> None:
    """
    Plot training curves (loss and accuracy) and save them as PNG images.

    Args:
    - hist (dict): A dictionary containing training and validation
    metrics (e.g., loss and accuracy).
    - path (str): The path to save the plots (without extension).
    """
    # plot training curves
    epochs = range(1, len(hist['train_loss']) + 1)

    train_loss = [t.cpu().detach().numpy() for t in hist['train_loss']]
    val_loss = [t.cpu().detach().numpy() for t in hist['val_loss']]

    _, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].plot(epochs, train_loss, 'r-', label='Train')
    axes[0].plot(epochs, val_loss, 'b-', label='Evaluation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    plt.savefig(f'{path}_Loss.png')  # This saves the plot as a PNG image

    axes[1].plot(epochs, hist['train_acc'], 'r-', label='Train')
    axes[1].plot(epochs, hist['val_acc'], 'b-', label='Evaluation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Acc')
    axes[1].legend()
    plt.savefig(f'{path}_Accuracy.png')  # This saves the plot as a PNG image

    plt.show()
