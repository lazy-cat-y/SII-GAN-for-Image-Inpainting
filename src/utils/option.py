import argparse

parser = argparse.ArgumentParser(description='SII-GAN model for image inpainting')

# data source
parser.add_argument('--dir_image', type=str, default='../dataset',
                    help='image dataset directory')
parser.add_argument('--dir_mask', type=str, default='../dataset',
                    help='mask dataset directory')
parser.add_argument('--data_train', type=str, default='train_image',
                    help='data name used for training')
parser.add_argument('--data_test', type=str, default='test_image',
                    help='data name used for testing')
parser.add_argument('--image_size', type=int, default=512,
                    help='image size uses during training')
parser.add_argument('--mask_type', type=str, default='pconv',
                    help='mask used during training')

# model
parser.add_argument('--model', type=str, default='siigan',
                    help='model name')
parser.add_argument('--block_num', type=str, default=10,
                    help='number of BDC block')
parser.add_argument('--rates', type=str, default='1+2+4+8',
                    help='dilation rates used in BDC block')
parser.add_argument('--gan_type', type=str, default='sngan',
                    help='discriminator types')

# hardware
parser.add_argument('--seed', type=int, default=2021,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=1,
                    help='number of workers used in data loader')

# optimization specifications 
parser.add_argument('--lrg', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lrd', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 in optimier')

# loss specifications 
parser.add_argument('--rec_loss', type=str, default='1*L1+250*Style+0.1*Perceptual',
                    help='losses for reconstruction')
parser.add_argument('--adv_weight', type=float, default=0.01,
                    help='loss weight for adversarial loss')

parser.add_argument('--iterations', type=int, default=1e6,
                    help='the number of iterations for training')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size in each mini-batch')
parser.add_argument('--port', type=int, default=22334,
                    help='tcp port for distributed training')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous iteration')

# log specifications 
parser.add_argument('--print_every', type=int, default=10,
                    help='frequency for updating progress bar')
parser.add_argument('--save_every', type=int, default=1e4,
                    help='frequency for saving models')
parser.add_argument('--save_dir', type=str, default='../experiments',
                    help='directory for saving models and logs')
parser.add_argument('--tensorboard', action='store_true',
                    help='default: false, since it will slow training. use it for debugging')

# test programme
parser.add_argument('--pre_train', type=str,
                    default='../experiments/siigan_train_image_pconv512/G00190000.pt',
                    help='pre-train model path')
parser.add_argument('--dir_test_image', type=str,
                    default='../inputs/test/image',
                    help='image test dataset directory')
parser.add_argument('--dir_test_mask', type=str,
                    default='../inputs/test/mask',
                    help='mask test dataset directory')
parser.add_argument('--outputs', type=str,
                    default='../outputs/',
                    help='outputs path')


args = parser.parse_args()
args.iterations = int(args.iterations)

args.rates = list(map(int, list(args.rates.split('+'))))

losses = list(args.rec_loss.split('+'))
args.rec_loss = {}
for lo in losses:
    weight, name = lo.split('*')
    args.rec_loss[name] = float(weight)
