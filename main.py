import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import torch
import sys
import os.path as osp
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloaders.dataset import VideoDataset
from args import argument_parser
from dataloaders.data_cfg import ds_cfg
import models
from utils.generaltools import set_random_seed
from utils.iotools import mkdir_if_missing
from utils.loggers import Logger

parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    set_random_seed(args.seed)
    dataset = args.source_names[0]
    epochs = args.max_epoch
    modelName = args.arch
    lr = args.lr
    save_dir = args.save_dir + '/' + dataset
    today = str(datetime.today().date())
    saveName = modelName + '_' + dataset + '_' + today
    root = args.root
    mkdir_if_missing(save_dir + "/models")

    log_name = "_log_test.txt" if args.evaluate else "_log_train.txt"
    sys.stdout = Logger(osp.join(save_dir, saveName + log_name))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    model = models.init_model(
        name=args.arch,
        num_classes=ds_cfg[dataset]['num_classes'],
        pretrained_model=args.pretrained_model
    )
    model.to(device)

    now = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("***********************************************")
    print(f"Running Time: {now}")
    print(f"==========\nArgs:{args}\n==========")

    train_params = [{'params': models.get_1x_lr_params(model), 'lr': lr},
                    {'params': models.get_10x_lr_params(model), 'lr': lr * 10}]

    batch_size = args.train_batch_size
    frame = args.frame
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=frame, root=root),
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=args.workers)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=frame, root=root),
                                batch_size=batch_size,
                                num_workers=args.workers)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=frame, root=root),
                                 batch_size=batch_size,
                                 num_workers=args.workers)
    print("train len:", len(train_dataloader))
    print("val   len:", len(val_dataloader))
    print("test  len:", len(test_dataloader))

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize,
                                          gamma=args.gamma)  # the scheduler divides the lr by 10 every 10 epochs

    if args.resume == '':
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            args.resume,
            map_location=device)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            args.resume))
        print("resume epoch: ", checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion.to(device)

    log_dir = os.path.join(save_dir, 'summary', today)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    for epoch in range(args.start_epoch, epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                model.train()
            else:
                model.eval()

            for inputs, labels in trainval_loaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch + 1)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch + 1)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch + 1)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch + 1)

            print("[{}] Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch + 1, epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(int(stop_time - start_time)) + "\n")

        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_' + str(epoch + 1) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_' + str(epoch + 1) + '.pth.tar')))

        if args.evaluate or (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch + 1)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch + 1)

            print("[test] Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(int(stop_time - start_time)) + "\n")

    writer.close()


if __name__ == "__main__":
    main()
