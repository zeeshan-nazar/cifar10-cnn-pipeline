import os
import torch
from torchvision.datasets import CIFAR10


RAW_DIR = "data/raw"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Downloading CIFAR-10...")

    train_dataset = CIFAR10(
        root=RAW_DIR,
        train=True,
        download=True
    )

    test_dataset = CIFAR10(
        root=RAW_DIR,
        train=False,
        download=True
    )

    # Convert PIL images to tensors
    train_images = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
    train_labels = torch.tensor(train_dataset.targets)

    test_images = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
    test_labels = torch.tensor(test_dataset.targets)

    torch.save(
        {
            "images": train_images,
            "labels": train_labels
        },
        f"{RAW_DIR}/train.pt"
    )

    torch.save(
        {
            "images": test_images,
            "labels": test_labels
        },
        f"{RAW_DIR}/test.pt"
    )

    print("Raw training data saved.")
    print("Raw testing data saved.")


if __name__ == "__main__":
    main()