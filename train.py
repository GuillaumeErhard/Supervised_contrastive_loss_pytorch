import argparse
import math
import numpy as np
import os
import torch

from torch.backends import cudnn
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from utils import progress_bar
from loss.spc import SupervisedContrastiveLoss
from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform

from models.resnet_contrastive import get_resnet_contrastive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet20",
        choices=[
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
        ],
        help="Model to use",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="dataset name",
    )
    parser.add_argument(
        "--training_mode",
        default="contrastive",
        choices=["contrastive", "cross-entropy"],
        help="Type of training use either a two steps contrastive then cross-entropy or \
                         just cross-entropy",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument("--temperature", default=0.07, type=float, help="Constant for loss no thorough ")
    parser.add_argument("--auto-augment", default=False, type=bool)

    parser.add_argument("--n_epochs_contrastive", default=500, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=100, type=int)

    parser.add_argument("--lr_contrastive", default=1e-1, type=float)
    parser.add_argument("--lr_cross_entropy", default=5e-2, type=float)

    parser.add_argument("--cosine", default=True, type=bool, help="Check this to use cosine annealing instead of ")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")
    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=[150, 300, 500],
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, mode, args):
    """

    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    if mode == "contrastive":
        lr = args.lr_contrastive
        n_epochs = args.n_epochs_contrastive
    elif mode == "cross_entropy":
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
    else:
        raise ValueError("Mode %s unknown" % mode)

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
    else:
        n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if n_steps_passed > 0:
            lr = lr * (args.lr_decay_rate ** n_steps_passed)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_contrastive(model, train_loader, criterion, optimizer, writer, args):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return: None
    """
    model.train()
    best_loss = float("inf")

    for epoch in range(args.n_epochs_contrastive):
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_contrastive))

        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = torch.cat(inputs)
            targets = targets.repeat(2)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            projections = model.forward_constrative(inputs)
            loss = criterion(projections, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar(
                "Loss train | Supervised Contrastive",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f " % (train_loss / (batch_idx + 1)),
            )

        avg_loss = train_loss / (batch_idx + 1)
        # Only check every 10 epochs otherwise you will always save
        if epoch % 10 == 0:
            if (train_loss / (batch_idx + 1)) < best_loss:
                print("Saving..")
                state = {
                    "net": model.state_dict(),
                    "avg_loss": avg_loss,
                    "epoch": epoch,
                }
                if not os.path.isdir("checkpoint"):
                    os.mkdir("checkpoint")
                torch.save(state, "./checkpoint/ckpt_contrastive.pth")
                best_loss = avg_loss

        adjust_learning_rate(optimizer, epoch, mode="contrastive", args=args)


def train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total_batch = targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            total += total_batch
            correct += correct_batch

            writer.add_scalar(
                "Loss train | Cross Entropy",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            writer.add_scalar(
                "Accuracy train | Cross Entropy",
                correct_batch / total_batch,
                epoch * len(train_loader) + batch_idx,
            )
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

        validation(epoch, model, test_loader, criterion, writer, args)

        adjust_learning_rate(optimizer, epoch, mode='cross_entropy', args=args)
    print("Finished Training")


def validation(epoch, model, test_loader, criterion, writer, args):
    """

    :param epoch: int
    :param model: torch.nn.Module, Model
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module, Loss
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt_cross_entropy.pth")
        args.best_acc = acc


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 10

    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_set = datasets.CIFAR100(root="~/data", train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = datasets.CIFAR100(root="~/data", train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 100

    model = get_resnet_contrastive(args.model, num_classes)
    model = model.to(args.device)

    cudnn.benchmark = True

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    writer = SummaryWriter("logs")

    if args.training_mode == "contrastive":
        train_contrastive_transform = DuplicateSampleTransform(transform_train)
        if args.dataset == "cifar10":
            train_set_contrastive = datasets.CIFAR10(
                root="~/data",
                train=True,
                download=True,
                transform=train_contrastive_transform,
            )
        elif args.dataset == "cifar100":
            train_set_contrastive = datasets.CIFAR100(
                root="~/data",
                train=True,
                download=True,
                transform=train_contrastive_transform,
            )

        train_loader_contrastive = torch.utils.data.DataLoader(
            train_set_contrastive,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        model = model.to(args.device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_contrastive,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        criterion = SupervisedContrastiveLoss(temperature=args.temperature)
        criterion.to(args.device)
        train_contrastive(model, train_loader_contrastive, criterion, optimizer, writer, args)

        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("./checkpoint/ckpt_contrastive.pth")
        model.load_state_dict(checkpoint["net"])

        model.freeze_projection()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_cross_entropy,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()
        criterion.to(args.device)

        args.best_acc = 0.0
        train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_cross_entropy,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(args.device)

        args.best_acc = 0.0
        train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args)


if __name__ == "__main__":
    main()
