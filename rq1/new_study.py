import math
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torchvision

from copy import deepcopy
from pathlib import Path
from pipeline.Dataloader.dataloader_factory import DataLoaderFactory
from rq1.parser import custom_argparse
from tqdm import tqdm
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class EndLayer(nn.Module):
    def __init__(self, in_features, normalized=False):
        super(EndLayer, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        self.normalized = normalized

    def forward(self, batch):
        output = self.fc(batch)
        if self.normalized:
            # if the data has been normalized from -1 to 1, then tanh is a better fit because that is its natural output
            output = torch.tanh(output)
        else:
            # if the data has not been normalized, then 2*atan is a natural fit bc it has output -pi to pi
            output = torch.atan(output)
            output = torch.multiply(output, 2)
        return output


def get_model(normalized=False, load_path: Path = None, eval=False):
    resnet34 = torchvision.models.resnet34(pretrained=True)
    resnet34.fc = EndLayer(resnet34.fc.in_features, normalized=normalized)
    resnet34 = resnet34.to(DEVICE)
    if load_path is not None:
        resnet34.load_state_dict(torch.load(load_path))
    if eval:
        resnet34.eval()
    return resnet34


def leave_one_out(dataloader, towns, complement=False):
    train_img_path_list = []
    train_label_list = []
    test_img_path_list = []
    test_label_list = []
    for img_path, label in zip(dataloader.img_list, dataloader.label_list):
        town_flag = True
        for town in towns:
            if town in img_path._parts:
                train_img_path_list.append(img_path)
                train_label_list.append(label)
                town_flag = False
                break
        if town_flag:
            test_img_path_list.append(img_path)
            test_label_list.append(label)

    train_dataloader = deepcopy(dataloader)
    test_dataloader = deepcopy(dataloader)

    train_dataloader.img_list = train_img_path_list if not complement else test_img_path_list
    train_dataloader.label_list = train_label_list if not complement else test_label_list

    test_dataloader.img_list = test_img_path_list if not complement else train_img_path_list
    test_dataloader.label_list = test_label_list if not complement else train_label_list

    return train_dataloader, test_dataloader


def leave_x_percent_out(dataloader, percent):
    df = pd.DataFrame(columns=['town', 'full_path', 'label'])
    df['town'] = [img_path.relative_to(dataloader.input_path).parts[0] for img_path in dataloader.img_list]
    df['full_path'] = [str(img_path) for img_path in dataloader.img_list]
    df['label'] = dataloader.label_list
    train_df = pd.DataFrame(columns=['town', 'full_path', 'label'])
    test_df = pd.DataFrame(columns=['town', 'full_path', 'label'])
    for town in df['town'].unique():
        split_n = math.ceil(len(df[df['town'] == town]) * percent)
        train_df = train_df.append(df[df['town'] == town][:split_n])
        test_df = test_df.append(df[df['town'] == town][split_n:])

    train_dataloader = deepcopy(dataloader)
    train_dataloader.img_list = [Path(p) for p in train_df['full_path'].tolist()]
    train_dataloader.label_list = train_df['label'].tolist()

    test_dataloader = deepcopy(dataloader)
    test_dataloader.img_list = [Path(p) for p in test_df['full_path'].tolist()]
    test_dataloader.label_list = test_df['label'].tolist()

    return train_dataloader, test_dataloader


def save_pred_in_file(trn_dl, tst_dl, trn_pred, tst_pred, csv_path):
    df = pd.DataFrame(columns=['split', 'town', 'filename', 'label', 'pred'])

    for split_name, split_df, split_pred in [('train', trn_dl, trn_pred), ('test', tst_dl, tst_pred)]:
        aux_df = pd.DataFrame(columns=['split', 'town', 'filename', 'label', 'pred'])
        aux_df['split'] = [split_name] * len(split_pred)
        aux_df['town'] = [img_path.relative_to(split_df.input_path).parts[0] for img_path in split_df.img_list]
        aux_df['filename'] = [str(img_path.relative_to(split_df.input_path).parts[1]) for img_path in split_df.img_list]
        aux_df['label'] = split_df.label_list
        aux_df['pred'] = split_pred

        df = df.append(aux_df)

    df.to_csv(csv_path, index=False)

def log_results_in_file(epoch, trn_loss, lr, csv_path):
    if not csv_path.exists():
        df = pd.DataFrame(columns=['epoch', 'trn_loss', 'lr'])
    else:
        df = pd.read_csv(csv_path)
    df = df.append({'epoch': epoch, 'trn_loss': trn_loss, 'lr': lr}, ignore_index=True)
    df.to_csv(csv_path, index=False)


def train(model, train_dataloader, optimizer, criterion):
    model.train()
    trn_losses = []
    trn_pred = []

    for x, y in tqdm(train_dataloader):
        x = x.float().to(DEVICE)
        y = y.float().to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        output = model(x)
        trn_pred.extend(output.cpu().squeeze().tolist())
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss = loss.data.item()
        trn_losses.append(loss)
    return np.mean(trn_losses), trn_pred


def test(model, test_dataloader, criterion):
    model.eval()
    test_losses = []
    test_pred = []

    for x, y in tqdm(test_dataloader):
        x = x.float().to(DEVICE)
        y = y.float().to(DEVICE).unsqueeze(1)
        with torch.no_grad():
            output = model(x)
            test_pred.extend(output.cpu().squeeze().tolist())
            loss = criterion(output, y)
            loss = loss.data.item()
            test_losses.append(loss)
    return np.mean(test_losses), test_pred


def main(arg_string):
    args = custom_argparse(arg_string)

    # Check that split_town or split_percent is specified
    assert args.split_town is not None or args.split_percent is not None, 'Either split_town or split_percent must be specified'
    # Check that split_town and split_percent are not both specified
    assert not (args.split_town is not None and args.split_percent is not None), 'split_town and split_percent cannot both be specified'
    # Check that split_town is set if town_complement is set
    assert not (args.town_complement and args.split_town is None), 'split_town must be specified if town_complement is set'

    dataloader = DataLoaderFactory(args.dataset_type, args.dataset_image_path, label_path=args.dataset_label_path,
                                   set_splits=None, loader_type='Training', shuffle=False, batch_size=args.batch_size,
                                   max_steering=args.max_steering, normalize_labels=args.normalize_labels)

    # Create train and test partitions based on split type
    if args.split_town is not None:
        train_dl, test_dl = leave_one_out(dataloader, args.split_town, args.town_complement)
    else:
        train_dl, test_dl = leave_x_percent_out(dataloader, args.split_percent)

    # Define augmentations
    augmentation = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAffine(degrees=0, translate=(5.0/320, 0))
    ])

    # Get torch dataloader for faster training
    train_torch_dataloader = train_dl.get_torch_dataloader(num_workers=args.num_workers, transformation=augmentation)
    test_torch_dataloader = test_dl.get_torch_dataloader(num_workers=args.num_workers)

    # Instantiate model
    resnet34 = get_model(normalized=args.normalize_labels)

    # Initialize training params
    optimizer = torch.optim.Adam(
        params=resnet34.parameters(),
        lr=0.0001
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=1e-4) # Default values

    total_time_start = time.perf_counter_ns()

    # Train log file
    if args.split_town is not None:
        if not args.town_complement:
            log_path = args.output_path / f"{args.split_town}_train_log.csv"
        else:
            log_path = args.output_path / f"{args.split_town}_complement_train_log.csv"
    else:
        log_path = args.output_path / f"{args.split_percent}_percent_train_log.csv"

    # Iterate over epochs
    for epoch in range(500):
        trn_loss, trn_pred = train(resnet34, train_torch_dataloader, optimizer, criterion)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1}; trn_loss {trn_loss:.6f}; lr {lr:.0e}')
        log_results_in_file(epoch+1, trn_loss, lr, log_path)

        scheduler.step(trn_loss)
        lr = optimizer.param_groups[0]['lr']

        # Evaluate every 25 epochs
        if (epoch+1) % 25 == 0 or lr <= 1e-6:
            test_loss, test_pred = test(resnet34, test_torch_dataloader, criterion)
            print(f'Epoch: {epoch+1}; test_loss {test_loss:.6f}; lr {lr:.0e}')

            # File names
            if args.split_town is not None:
                if not args.town_complement:
                    csv_path = args.output_path / f"{args.split_town}_e{epoch+1}_results.csv"
                    weight_path = args.output_path / f"{args.split_town}_e{epoch+1}_model.pt"
                else:
                    csv_path = args.output_path / f"{args.split_town}_complement_e{epoch+1}_results.csv"
                    weight_path = args.output_path / f"{args.split_town}_complement_e{epoch+1}_model.pt"
            else:
                csv_path = args.output_path / f"{args.split_percent}_percent_e{epoch+1}_results.csv"
                weight_path = args.output_path / f"{args.split_percent}_percent_e{epoch+1}_model.pt"

            # Save pred in csv
            save_pred_in_file(train_dl, test_dl, trn_pred, test_pred, csv_path)

            # Save model
            print('Saving the model...')
            resnet34 = resnet34.to(torch.device('cpu'))
            torch.save(resnet34.state_dict(), weight_path)
            resnet34 = resnet34.to(DEVICE)

        if lr <= 1e-6:
            print('Finished earlier, lr of 1e-6 reached!')
            break

    total_time_end = time.perf_counter_ns()
    total_time_spent = (total_time_end-total_time_start)*1e-9
    print(f"Total time spent for trainining: {total_time_spent}")


if __name__ == "__main__":
    main()
