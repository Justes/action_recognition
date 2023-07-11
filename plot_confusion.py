import numpy as np
import itertools
import matplotlib.pyplot as plt


def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Remove newlines from each line and store them in a list
    lines = [line.strip() for line in lines]
    return lines


m = 0
n = 51

file_name = 'conf_matrix.txt'
confusion_matrix = np.loadtxt(file_name, dtype=int, delimiter=',')
print(confusion_matrix.shape)
confusion_matrix = confusion_matrix[m:n, m:n]
num_classes = n
class_labels = read_txt_file('dataloaders/ucf_labels.txt')
class_labels = class_labels[m:n]

proportion = []
for i in confusion_matrix:
    for j in i:
        tmp = j / (np.sum(i))
        proportion.append(tmp)

proportion = np.array(proportion).reshape(num_classes, num_classes)

# turbo, plasma, gnuplot
plt.imshow(proportion, interpolation='nearest', cmap='turbo')
plt.title('HMDB51 Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_labels, size=4, rotation=90)
plt.yticks(tick_marks, class_labels, size=4)

thresh = proportion.max() / 2.
for i, j in itertools.product(range(num_classes), range(num_classes)):
    if confusion_matrix[i, j] != 0:
        plt.text(j, i, "",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="blue" if proportion[i, j] > thresh else "red")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
