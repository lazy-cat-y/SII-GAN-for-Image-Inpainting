import os
import importlib
from PIL import Image
from glob import glob

import torch
from torchvision.transforms import ToTensor
import numpy as np

from utils.option import args


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def test_worker(args, use_gpu=True):
    torch.device('cuda') if use_gpu else torch.device('cpu')

    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    image_paths = []
    for ext in ['.jpg', '.png']:
        image_paths.extend(glob(os.path.join(args.dir_test_image, '*'+ext)))
    image_paths.sort()
    mask_paths = sorted(glob(os.path.join(args.dir_test_mask, '*.png')))
    os.makedirs(args.outputs, exist_ok=True)

    for i, m in zip(image_paths, mask_paths):
        image = ToTensor()(Image.open(i).convert('RGB'))
        image = (image * 2. - 1.).unsqueeze(0)
        mask = ToTensor()(Image.open(m).convert('L'))
        mask = mask.unsqueeze(0)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask

        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_image = (1 - mask) * image + mask * pred_img
        image_name = os.path.basename(i).split('.')[0]
        postprocess(image_masked[0]).save(os.path.join(args.outputs, f'{image_name}_masked.png'))
        postprocess(pred_img[0]).save(os.path.join(args.outputs, f'{image_name}_pred.png'))
        postprocess(comp_image[0]).save(os.path.join(args.outputs, f'{image_name}_comp.png'))
        print(f'saving to {os.path.join(args.outputs, image_name)}')


if __name__ == '__main__':
    test_worker(args)
