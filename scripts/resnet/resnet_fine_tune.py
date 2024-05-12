import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from torch.cuda.amp import autocast, GradScaler
from collections import deque

from src.models.resnet18 import ResNet18
from src.dataset.dataset import AnimalDataset
from src.utils.transform import get_transform
from src.utils.log import log
from src.utils.plot_loss import plot_loss
from src.utils.plot_accuracy import plot_accuracy
from src.utils.plot_confusion_matrix import plot_confusion_matrix
from src.utils.save_misclassified import save_misclassified_images
from src.config.args import parser
from src.constants.constants import CLASS_NAMES

torch.cuda.empty_cache()


def train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, validation_accuracies, val_loader, args):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

        losses.append(loss.item())

        if batch_idx % args.plot_loss_every_n_iteration == 0 and batch_idx != 0:
            plot_loss(losses, f"loss_epoch_{epoch}_idx_{batch_idx}", args)

    validate(model, device, val_loader, epoch, loss_fn, validation_accuracies, args)


def validate(model, device, test_loader, epoch, loss_fn, accuracies, args):
    model.eval()
    test_loss = 0
    correct = 0

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch}, validation")

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    accuracies.append(100.0 * correct / len(test_loader.dataset))

    plot_accuracy(accuracies, f"accuracy_epoch_{epoch}", args)


def test(model, device, test_loader, epoch, loss_fn, args):
    model.eval()
    test_loss = 0
    correct = 0

    all_preds = []
    all_labels = []
    misclassified_examples = []

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch}, test")

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += (pred == target).sum().item()

            all_preds.append(pred)
            all_labels.append(target)

            # Identify misclassified examples
            misclassified_indices = pred != target
            misclassified_data = data[misclassified_indices]
            misclassified_targets = target[misclassified_indices]
            misclassified_preds = pred[misclassified_indices]

            for i in range(misclassified_data.size(0)):
                example = {
                    "data": misclassified_data[i],
                    "true_label": CLASS_NAMES[misclassified_targets[i].item()],
                    "predicted_label": CLASS_NAMES[misclassified_preds[i].item()],
                }
                misclassified_examples.append(example)

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )

    all_preds = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()

    plot_confusion_matrix(all_labels.numpy(), all_preds.numpy(), f"confusion_matrix_epoch_{epoch}", args)
    save_misclassified_images(misclassified_examples, f"results/{args.run_name}/misclassified/{epoch}/")


def main(args):
    full_dataset = AnimalDataset(root_dir=args.dataset_dir, transform=get_transform(resize=256, crop=224))

    # Splitting the dataset into train and test sets
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size * 2

    train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, test_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True
    )

    device = args.device

    # Load the pre-trained ResNet model
    """
    IMAGENET1K_V1: The weights are pre-trained on the ImageNet1K dataset.
    DEFAULT: DEFAULT = IMAGENET1K_V1 (I guess they are the same, not worth changing...)
    """

    resnet_version = args.resnet_version

    log(f"Loading pre-trained ResNet{resnet_version} model")

    if resnet_version == "18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif resnet_version == "34":
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif resnet_version == "50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Invalid model version")

    log(f"Loaded pre-trained ResNet{resnet_version} model")

    # Modify the last fully connected layer to match the number of classes
    last_layer_in_features = model.fc.in_features
    model.fc = nn.Linear(last_layer_in_features, len(CLASS_NAMES))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    # Only train the last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    losses = deque(maxlen=1000)

    # Test the model without any training.
    # test(model, device, test_loader, -1, loss_fn, args)

    validation_accuracies = []

    for epoch in range(1, args.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn, scaler, losses, validation_accuracies, validation_loader, args)
        test(model, device, test_loader, epoch, loss_fn, args)

        if args.save_checkpoints and epoch % args.save_checkpoints_epoch == 0:

            if len(validation_accuracies) > 0 and validation_accuracies[-1] >= max(validation_accuracies):
                torch.save(model.state_dict(), f"checkpoints/{args.run_name}/best_model.pth")
            else:
                torch.save(model.state_dict(), f"checkpoints/{args.run_name}/epoch_{epoch}.pth")

    if args.save_checkpoints:
        torch.save(model.state_dict(), f"checkpoints/{args.run_name}/final.pth")


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parser.parse_args()

    args.run_name = "fine_tune__" + args.run_name
    args.device = torch.device(args.device)
    log("Using device:", args.device)

    log("Using args:", args)
    os.makedirs(f"checkpoints/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/loss/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/conf/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/misclassified/", exist_ok=True)
    os.makedirs(f"results/{args.run_name}/accuracy/", exist_ok=True)
    main(args)
