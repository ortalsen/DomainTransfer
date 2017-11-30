import argparse
import os
import tensorflow as tf
from model import AutoDIAL
os.environ["CUDA_VISIBLE_DEVICES"]="7"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='ETH_OlegSim', help='path of the dataset')
parser.add_argument('--minVal', dest='minVal', default=40.0, help='lower clipping level')
parser.add_argument('--maxVal', dest='maxVal', default=400.0, help='upper clipping level')
parser.add_argument('--QuanFactor', dest='QuanFactor', default=10.0, help='quantization factor')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=6, help='# images in batch (n). Batch final size is 2n, n for source, n for target ')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--continue_training', dest='continue_training', type=bool, default=False, help='continue training from checkpoint')
parser.add_argument('--val_iter', dest='val_iter', default=0.25, help='# of iterations before validation as a part from an epoch')
parser.add_argument('--DS_FACTOR', dest='DS_FACTOR', type=int, default=2, help='down-sampling factor')
parser.add_argument('--kernel_size', dest='kernel_size', type=int, default=3, help='network kernels size')
parser.add_argument('--input_width', dest='input_width', type=int, default=256, help='size of input image')
parser.add_argument('--input_ch', dest='input_ch', type=int, default=1, help='# of input channels')
parser.add_argument('--output_ch', dest='output_ch', type=int, default=1, help='# of output channels')
parser.add_argument('--MAX_BIF', dest='MAX_BIF', type=int, default=5, help='max # of bifurcations of residual network')
parser.add_argument('--initial_depth', dest='initial_depth', type=int, default=8, help='# number of channels fro firt convolution layer')
parser.add_argument('--Lambda', dest='Lambda', type=float, default=0.1, help='weight of target entropy loss')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for ADAM optimizer')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint1', help='models are saved here')
parser.add_argument('--exp_name', dest='exp_name', default='initial_exp', help='descriptive name of the experiment')
parser.add_argument('--results_dir', dest='results_dir', default='./results1', help='log and training curve are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test1', help='test sample are saved here')
parser.add_argument('--domain', dest='domain', default='source', help='domain for test: source/target')


args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)


    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = AutoDIAL(sess, args)
        model.train() if args.phase == 'train' \
            else model.test()

if __name__ == '__main__':
    tf.app.run()