from __future__ import division
import json
import os
from tqdm import tqdm
import warnings
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from models.LGFINet import create_lgfi
from utils import *
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)
logger = logging.getLogger('mnist_AutoML')

tensorboard_out_dir = 'experiment/'

def main(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # visualization our loss
    train_writer = SummaryWriter(os.path.join(tensorboard_out_dir, 'train'))
    
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(512),
                                   transforms.CenterCrop(448),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder(root="CUB200_Small/train", transform=data_transform["train"]) # don't forget to change path to load data
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args["num_workers"])

    validate_dataset = datasets.ImageFolder(root="CUB200_Small/test", transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args["num_workers"])

    bird_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in bird_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('./datasets/CUB200/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    print("using {} images for training, {} images fot validation.".format(train_num, val_num))

    net = create_lgfi(img_size=(448, 448)).to(device) # our new network

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9) # change a little bit lr and weight_decay

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['num_workers'])
    print(args['best_pred'], args['start_epoch'])
    best_acc = 0.0

    for epoch in range(args['start_epoch'], args['epochs']):

        # # train
        net.train()
        for _, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch, args['epochs'], loss)

        # validate
        if epoch == 0 or epoch % args["test_frequency"] == 0 and epoch >= 0:
            net.eval()
            val_acc = 0.0
            train_acc = 0.0
            val_loss = 0.0
            train_loss = 0.0

            with torch.no_grad():
                for train_data in train_loader:
                    train_images, train_labels = train_data
                    train_outputs = net(train_images.to(device))
                    tmp_train_loss = loss_function(train_outputs, train_labels.to(device))
                    train_predict = torch.max(train_outputs, dim=1)[1]
                    train_acc += torch.eq(train_predict, train_labels.to(device)).sum().item()
                    train_loss += tmp_train_loss.item()
                    # train_bar.desc = "valid in train_dataset epoch[{}/{}]".format(epoch + 1, args['epochs'])
                for val_data in validate_loader:
                    val_images, val_labels = val_data
                    val_outputs = net(val_images.to(device))
                    tmp_val_loss = loss_function(val_outputs, val_labels.to(device))
                    val_predict = torch.max(val_outputs, dim=1)[1]
                    val_acc += torch.eq(val_predict, val_labels.to(device)).sum().item()
                    val_loss += tmp_val_loss.item()
                    # val_bar.desc = "valid in val_dataset epoch[{}/{}]".format(epoch + 1, args['epochs'])

            train_accurate = train_acc / train_num
            val_accurate = val_acc / val_num

            print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss:%.3f val_acc: %.3f'
                  % (epoch, train_loss / 188, train_accurate, val_loss / val_num, val_accurate))
            if (val_accurate > best_acc):
                best_acc = val_accurate
            is_best = best_acc > args['best_pred']
            args['best_pred'] = max(best_acc, args['best_pred'])
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': net.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])

            # this is for visualize loss and acc in tensorboard 
            train_writer.add_scalars('loss_part',
            {
                'train_loss' : train_loss / train_num,
                'val_loss' : val_loss / val_num
            }, epoch)

            train_writer.add_scalars('acc_part',
            {
                'train_acc' : train_accurate,
                'val_acc' : val_accurate
            }, epoch)


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)