import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import timeit
from torch.utils.data import DataLoader
from dataloaders.dataset import VideoDataset
from dataloaders.data_cfg import ds_cfg
import models
from utils.generaltools import set_random_seed


def plot_confusion_matrix(model, dataloader):
    """Plots the confusion matrix for a PyTorch model and data loader."""

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the model to evaluation mode
    model.eval()

    class_labels = dataloader.dataset.classes
    num_classes = len(class_labels)

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    criterion = nn.CrossEntropyLoss()
    test_size = len(dataloader.dataset)
    epoch = 0
    epochs = 1

    # Iterate over the data and accumulate the confusion matrix
    with torch.no_grad():
        start_time = timeit.default_timer()
        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size

        print("[test] Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(int(stop_time - start_time)) + "s\n")

    # Plot the confusion matrix
    np.savetxt("conf_matrix.txt", confusion_matrix, fmt="%d", delimiter=",")
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        plt.text(j, i, format(int(confusion_matrix[i, j]), fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    dataset = 'hmdb51'
    resume = '../pretrained/c3d_hmdb51_2023-07-05_25.pth.tar'
    arch = 'c3d'

    workers = 2
    frame = 16
    root = '../datasets'
    batch_size = 32
    set_random_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = ds_cfg[dataset]['num_classes']

    model = models.init_model(
        name=arch,
        num_classes=num_classes
    )

    checkpoint = torch.load(resume, map_location=device)  # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(resume))
    print("resume epoch: ", checkpoint['epoch'])
    model.load_state_dict(checkpoint['state_dict'])

    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=frame, root=root),
                                 batch_size=batch_size,
                                 num_workers=workers)
    # print(len(test_dataloader.dataset.classes))
    # for inputs, labels in test_dataloader:
    #     print(labels)
        # print(test_dataloader.dataset.classes[labels])
    plot_confusion_matrix(model, test_dataloader)