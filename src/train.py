import os
import csv
import torch
import yaml

from torch import nn
from torch.utils.data import TensorDataset, DataLoader


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


class CNN(nn.Module):

    def __init__(self, num_filters, dropout_rate):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                num_filters,
                num_filters * 2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                num_filters * 2 * 8 * 8,
                256
            ),

            nn.ReLU(),

            nn.Dropout(dropout_rate),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    train_params = params["train"]

    num_filters = train_params["num_filters"]
    dropout_rate = train_params["dropout_rate"]
    learning_rate = train_params["learning_rate"]
    epochs = train_params["epochs"]
    batch_size = train_params["batch_size"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_data = torch.load(
        f"{PROCESSED_DIR}/train.pt"
    )

    val_data = torch.load(
        f"{PROCESSED_DIR}/val.pt"
    )

    train_dataset = TensorDataset(
        train_data["images"],
        train_data["labels"]
    )

    val_dataset = TensorDataset(
        val_data["images"],
        val_data["labels"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = CNN(
        num_filters=num_filters,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        history.append(
            [
                epoch + 1,
                train_loss / len(train_loader),
                train_accuracy,
                val_loss / len(val_loader),
                val_accuracy
            ]
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_filters": num_filters,
            "dropout_rate": dropout_rate
        },
        f"{MODEL_DIR}/model.pth"
    )

    with open(
        f"{MODEL_DIR}/history.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy"
            ]
        )

        writer.writerows(history)

    print("Model saved.")
    print("Training history saved.")


if __name__ == "__main__":
    main()