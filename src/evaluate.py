import json
import torch
import matplotlib.pyplot as plt
import yaml

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/model.pth"


class CNN(nn.Module):

    def __init__(self, num_filters, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                num_filters,
                num_filters * 2,
                3,
                padding=1
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * 2 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    train_params = params["train"]

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    num_filters = checkpoint["num_filters"]
    dropout_rate = checkpoint["dropout_rate"]

    model = CNN(
        num_filters,
        dropout_rate
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    test_data = torch.load(
        f"{PROCESSED_DIR}/test.pt"
    )

    dataset = TensorDataset(
        test_data["images"],
        test_data["labels"]
    )

    loader = DataLoader(
        dataset,
        batch_size=train_params["batch_size"]
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.numpy()
            )

    test_loss = total_loss / len(loader)
    test_accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()
    plt.title("CIFAR-10 Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

    with open(
        "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()