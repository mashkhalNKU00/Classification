import argparse

parser = argparse.ArgumentParser(description='Fine_Grained_Image_Classify')

parser.add_argument('--save_path', type=str, default= './save_files/CUB200', help='save checkpoint directory')
parser.add_argument('--num_workers', type=int, default=1, help='load data workers')
parser.add_argument('--test_frequency', type=int, default=5, help='load data workers')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch for training')
parser.add_argument('--epochs', type=int, default=300, help='end epoch for training')
parser.add_argument('--pre', type=str, default=None, help='pre-trained model directory') # change this into none
parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--best_pred', type=int, default=0, help='best pred')

args = parser.parse_args()
return_args = parser.parse_args()
