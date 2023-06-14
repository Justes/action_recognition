import numpy as np
import itertools
import matplotlib.pyplot as plt


def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Remove newlines from each line and store them in a list
    lines = [line.strip() for line in lines]
    return lines


m = 75
n = 101

file_name = 'conf_matrix.txt'
confusion_matrix = np.loadtxt(file_name, dtype=int, delimiter=',')
print(confusion_matrix.shape)
confusion_matrix = confusion_matrix[m:n, m:n]
num_classes = 26
class_labels = read_txt_file('dataloaders/ucf_labels.txt')
class_labels = class_labels[m:n]

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('UCF101 Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_labels, rotation=90)
plt.yticks(tick_marks, class_labels)

fmt = 'd'
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(num_classes), range(num_classes)):
    if confusion_matrix[i, j] != 0:
        plt.text(j, i, format(int(confusion_matrix[i, j]), fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
