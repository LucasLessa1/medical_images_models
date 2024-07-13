# -*- coding: utf-8 -*-
"""
# Imports
"""

from model_preprocessing import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image
from trainer import *
import PIL
import random
import numpy as np
from tool_preprocessing import *
from trainer import *
from model_metrics import *
from models import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image
from model_preprocessing import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""# Preprocessing

In this notebook, labels are initially considered as categorical.

## Manual Part

If the images are organized in the folders of each label, the following flag must be True
"""


flag_folder_sep = True

if flag_folder_sep:

    base_path = './COVID_Dataset_original/'

    results_path = f"./Resultados_COVID/"
    os.makedirs(results_path)
else:
    base_path = 'C:/Users/lucas/OneDrive - unb.br/Documents/UnB/Semestres-ENE/TCC/The HAM10000 dataset'
    results_path = f"C:/Users/Lucas/medical_images_models/results_HAM"

if flag_folder_sep :
    label_column = 'label'
    train_df, test_df, val_df  = make_dataset_by_folder(
        base_path=base_path,
        label_column=label_column)

else:

    path_train_df = f'{base_path}/HAM10000_metadata'
    path_test_df = f'{base_path}/test.csv'

    path_train = f"{base_path}/treino"
    path_test = f"{base_path}/test"

    paths_image = [path_train, path_test]
    paths_df = [path_train_df, path_test_df]
    label_column = 'dx'

    train_df, test_df, val_df = make_dataset_by_df(
        paths_image,
        paths_df,
        label_column=label_column)

"""## Analysis

### Train
"""

train_df = check_images_existence(train_df,
                                  path_column='path')

train_df.to_csv("teste.csv")

image_analysis_train = image_analysis(train_df)

dict_train_qntd = get_label_counts_and_print(train_df,
                                             label_column=label_column)
shapes_train = analyze_image_shapes(train_df,
                                    min_shape=(800, 800),
                                    path_column='path')

dict_train_qntd

"""### Test"""

test_df = check_images_existence(test_df,
                                 path_column='path')

image_analysis_test = image_analysis(test_df)

dict_test_qntd = get_label_counts_and_print(test_df,
                                            label_column=label_column)
shapes_test = analyze_image_shapes(test_df,
                                   min_shape=(300, 300),
                                   path_column='path')

"""### Validation"""

val_df = check_images_existence(val_df,
                                path_column='path')

image_analysis_val = image_analysis(val_df)

dict_val_qntd = get_label_counts_and_print(val_df,
                                           label_column=label_column)
shapes_val = analyze_image_shapes(val_df,
                                  min_shape=(461, 601),
                                  path_column='path')

"""## Model Preparation"""



"""Passar de categorial para binário

Pesos para a loss

### Categorial to number
"""

labels_dict = labels2dict(train_df,
                          label_column)
labels_dict

train_label, test_label, val_label = dflabel2number([
    train_df,
    test_df,
    val_df], labels_dict, label_column)

"""### Weights"""

if len(labels_dict) == 1:
    weights = calculate_weights(train_df,
                                labels_dict,
                                dict_train_qntd)
    weights = max(weights)
else:
    weights = calculate_weights(train_df,
                                labels_dict,
                                dict_train_qntd)
    print(weights)

"""# Model"""




class CT_Dataset(Dataset):
    """
    Custom dataset class for CT images.

    Args:
        img_path (list): List containing paths to the
        CT images.
        img_labels (list): List containing labels for
        the CT images.
        channels (int): Number of channels in the images.
        Must be either 1 or 3.
        img_transforms (torchvision.transforms.Compose, optional):
        Transformations to apply to the images.
            Defaults to None.
    """
    def __init__(self,
                 img_path: list,
                 img_labels: list,
                 channels: int,
                 img_transforms=None) -> None:
        self.img_path = img_path
        self.img_labels = torch.Tensor(img_labels)
        if channels == 1:
            self.transforms = transforms.Compose((
                [transforms.Grayscale(),
                 transforms.ToTensor()]))
        elif channels == 3:
            self.transforms = transforms.Compose((
                [transforms.ToTensor()]))
        else:
            self.transforms = img_transforms

    def __getitem__(self,
                    index: int
                    ) -> tuple:
        """
        Retrieves an image and its corresponding label.

        Args:
            index (int): Index of the image and label
            to retrieve.

        Returns:
            tuple: Tuple containing the image and its
            corresponding label.
        """
        # Load image
        cur_path = self.img_path[index]
        cur_img = PIL.Image.open(cur_path).convert('RGB')
        cur_img = self.transforms(cur_img)

        return cur_img, self.img_labels[index]

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.img_path)

"""## GPU"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

torch.cuda.empty_cache()

print("Current GPU memory usage:",
      torch.cuda.memory_allocated() / (1024 ** 2), "MB")

print("Max GPU memory usage:",
      torch.cuda.max_memory_allocated() / (1024 ** 2), "MB")

"""## Random Seed"""

random_seed = 124
np.random.seed(random_seed)

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

"""## Training"""

try:
    mean_R = image_analysis_val['channel_statistics']['R']['average']
    mean_G = image_analysis_val['channel_statistics']['G']['average']
    mean_B = image_analysis_val['channel_statistics']['B']['average']
    channels = 1 if mean_R == mean_G == mean_B else 3

except KeyError:
    channels = image_analysis_val['channels']

train_dataset = CT_Dataset(img_path=np.array(train_df['path']),
                           img_labels=np.array(train_label),
                           channels=channels)

val_dataset = CT_Dataset(img_path=np.array(val_df['path']),
                         img_labels=np.array(val_label),
                         channels=channels)

test_dataset = CT_Dataset(img_path=np.array(test_df['path']),
                          img_labels=np.array(test_label),
                          channels=channels)



batch_size = 32
Epochs = 20

# model_kernel = VGG16(num_classes=len(labels_dict),
#                      input_channels=channels)

model_kernel = ResNet50(num_classes=len(labels_dict),
                        input_channels=channels)

# model_kernel = ResNet101(num_classes=len(labels_dict),
#                          input_channels=channels)

# model_kernel = EfficientNetB0(num_classes=len(labels_dict),
#                               input_channels=channels)

# model_kernel = EfficientNetB4(num_classes=len(labels_dict),
#                               input_channels=channels)

# model_kernel = EfficientNetB7(num_classes=len(labels_dict),
#                               input_channels=channels)


trainer = ModelTrainer(model_kernel,
                       device,
                       weights,
                       labels_dict,
                       train_dataset,
                       val_dataset,
                       test_dataset,
                       batch_size=batch_size,
                       epochs=Epochs)

trainer.loader()
trainer.loss_function()
trainer.optimizer_step()
print("Training Start:")
for epoch in range(Epochs):
    trainer.model.train()

    trainer.train_loss = 0
    trainer.train_acc = 0

    trainer.train()
    trainer.validate()
    history = trainer.loss_acc()

    print(f"Epoch:{epoch + 1} / {Epochs},"
          f" lr: {trainer.optimizer.param_groups[0]['lr']:.5f}"
          f" train loss:{trainer.train_loss:.5f}, "
          f"train acc: {trainer.train_acc:.5f}, "
          f"valid loss:{trainer.val_loss:.5f}, "
          f"valid acc:{trainer.val_acc:.5f}")

    # Update the best model if validation loss is the lowest so far
    if trainer.val_loss < trainer.best_val_loss:
        trainer.best_val_loss = trainer.val_loss
        trainer.best_model_state = trainer.model.state_dict()

    print(f'The best val loss is {trainer.best_val_loss}.\n')

    # Load the best model state
    if trainer.best_model_state is not None:
        trainer.model.load_state_dict(trainer.best_model_state)
    model = trainer.model

trainer.test()
metrics_df = trainer.metrics()

metrics_df = trainer.metrics()
metrics_df = metrics_df.applymap((
    lambda x: str(x).replace('.', ',')))

"""# Metrics"""



metrics_df

metrics_df.to_csv((
    f"{results_path}/Model_{model.get_name()}"
    f"__Epoch_{Epochs}__Batch_{batch_size}__"
    f"Accuracy_{metrics_df['Accuracy'][0]}.csv"), index=False)

plot_metrics(history,
             path=results_path)

"""## Plot Images - True Predicted"""

inverted_labels_dict = {value: key for key, value in labels_dict.items()}
inverted_labels_dict

plot_image_pred_true(model,
                     test_dataset,
                     device,
                     inverted_labels_dict,
                     num_images_to_plot=20,
                     plot_images=True)

"""# Save Model"""

torch.save(model.state_dict(),
           (f"{results_path}/Model_{model.get_name()}"
            f"__Epoch_{Epochs}__Batch_{batch_size}__"
            f"Accuracy_{metrics_df['Accuracy'][0]}.pth"))