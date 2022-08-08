import argparse
from PIL import Image
from pathlib import Path
import numpy as np
import math
import tqdm
import torch
import captcha_model

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume-training", action="store_true")
    return parser.parse_args()


def get_manually_classified(input_dir: Path):
    manual_classified = []
    for file in tqdm.tqdm(input_dir.iterdir()):
        if not file.is_file():
            continue
        label = file.stem
        img = Image.open(file).convert("L")
        yield (label, img)


def split_letters(image, letter_count):
    w, h = image.size
    part_width = w / letter_count
    parts = []
    for i in range(letter_count):
        yield image.crop((i * part_width, 0, i * part_width + part_width, h))


def get_class(letter):
    return torch.tensor(captcha_model.CLASSES.index(letter))


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)


def get_letters_separate(device, input_dir):
    for label, image in get_manually_classified(input_dir):
        for letter, img in zip(
            label, split_letters(image, letter_count=captcha_model.LETTER_COUNT)
        ):
            tensor = transform(img)
            yield tensor.to(device), get_class(letter).to(device)


class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, start_perc, end_perc, device, input_dir):
        super().__init__()
        data = list(get_letters_separate(device, input_dir))
        start = math.floor(len(data) * start_perc)
        end = math.floor(len(data) * end_perc)
        self.data = data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(net, train_loader, output_dir, epoch):
    net.train()
    for (data, target) in tqdm.tqdm(
        train_loader, position=1, leave=False, desc="Batch"
    ):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), output_dir / "model.pth")
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")


def test(net, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().cpu()
    test_loss /= len(test_loader.dataset)
    test_len = len(test_loader.dataset)
    return "Test: loss {:.2f}, Acc: {:.1f}%".format(
        test_loss,
        100.0 * correct / test_len,
    )


if __name__ == "__main__":
    # Training settings
    n_epochs = 200
    batch_size_train = 128
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 1

    args = parse_cmdline()
    device = captcha_model.get_device()
    net = captcha_model.Net().to(device)
    args.output.mkdir(exist_ok=True)

    torch.manual_seed(random_seed)

    test_loader = torch.utils.data.DataLoader(
        CaptchaDataset(start_perc=0, end_perc=0.1, device=device, input_dir=args.input),
        batch_size=batch_size_test,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        CaptchaDataset(start_perc=0.1, end_perc=1, device=device, input_dir=args.input),
        batch_size=batch_size_train,
        shuffle=True,
    )

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    if args.resume_training:
        net.load_state_dict(torch.load(args.output / "model.pth"))
        optimizer.load_state_dict(torch.load(args.output / "optimizer.pth"))

    test(net, test_loader)
    for epoch in (bar := tqdm.tqdm(range(1, n_epochs + 1), position=0)):
        train(net, train_loader, args.output, epoch)
        test_status = test(net, test_loader)
        bar.set_description(test_status)
