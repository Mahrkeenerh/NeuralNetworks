
with open("data/fashion_mnist_train_labels.csv") as f:
    train_labels = f.readlines()

with open("data/fashion_mnist_test_labels.csv") as f:
    test_labels = f.readlines()

with open("train_predictions.csv") as f:
    train_predictions = f.readlines()

with open("test_predictions.csv") as f:
    test_predictions = f.readlines()

train_correct = 0

for i in range(len(train_labels)):
    if train_labels[i] == train_predictions[i]:
        train_correct += 1

test_correct = 0

for i in range(len(test_labels)):
    if test_labels[i] == test_predictions[i]:
        test_correct += 1

print("Train accuracy: ", train_correct / len(train_labels))
print("Test accuracy: ", test_correct / len(test_labels))
