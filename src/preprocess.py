import os
import torch
import yaml


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def main():

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    preprocess_params = params["preprocess"]

    val_size = preprocess_params["val_size"]
    seed = preprocess_params["seed"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_data = torch.load(f"{RAW_DIR}/train.pt")
    test_data = torch.load(f"{RAW_DIR}/test.pt")

    train_images = train_data["images"].float() / 255.0
    train_labels = train_data["labels"]

    test_images = test_data["images"].float() / 255.0
    test_labels = test_data["labels"]

    # CIFAR-10 mean and standard deviation
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    #train_images = (train_images - mean) / std
    #test_images = (test_images - mean) / std
    train_images = (train_images - 0.5) / 0.5
    test_images = (test_images - 0.5) / 0.5

    # Shuffle and create validation split
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_images),
        generator=generator
    )

    val_count = int(len(train_images) * val_size)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    processed_train = {
        "images": train_images[train_indices],
        "labels": train_labels[train_indices]
    }

    processed_val = {
        "images": train_images[val_indices],
        "labels": train_labels[val_indices]
    }

    processed_test = {
        "images": test_images,
        "labels": test_labels
    }

    torch.save(processed_train, f"{PROCESSED_DIR}/train.pt")
    torch.save(processed_val, f"{PROCESSED_DIR}/val.pt")
    torch.save(processed_test, f"{PROCESSED_DIR}/test.pt")

    print("Preprocessing completed.")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_images)}")


if __name__ == "__main__":
    main()