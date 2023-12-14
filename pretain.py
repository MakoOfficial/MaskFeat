import torch
from collections import OrderedDict
from model import ViT
import argparse
from datasets import build_pretraining_dataset_for_MaskFeat
from torch import nn
from operators import HOGLayerC
import math
import sys
from torch.optim import AdamW

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--data_path', default="../archive/masked_crop/train", type=str, help='dataset path')
    parser.add_argument('--save_path', default="./output", type=str, help='save image path')
    parser.add_argument('--model_path', default="./checkpoint/masked_40.pth", type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='training in one batch')
    parser.add_argument('--epochs', default=1600, type=int,
                        help='total round')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    return parser.parse_args()

args = get_args()

def prepare_model(chkpt_dir):
    # build model
    model = ViT(
        num_classes=108
    )
    #   load checkpoint
    checkpoint_model = torch.load(chkpt_dir, map_location='cpu')['model']
    state_dict = model.state_dict()

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith("encoder."):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    print((new_dict.keys()))
    state_dict.update({k: v for k, v in new_dict.items() if k in state_dict.keys()})

    msg = model.load_state_dict(state_dict)
    print(msg)
    return model

chkpt_dir = './pretrain_mae_vit_base_mask_0.75_400e.pth'

model = prepare_model(chkpt_dir)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model = %s" % str(model))
print('number of params: {} M'.format(n_parameters / 1e6))

patch_size = model.patch_embed.patch_size
print(f"Patch size = {patch_size}")
args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
args.patch_size = patch_size

training_dataset = build_pretraining_dataset_for_MaskFeat(args)
training_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=args.batch_size
)

device = torch.device(args.device)
loss_func = nn.MSELoss(reduction='sum')
pool = 8
nbins = 9
cpb = 2
HogLayer = HOGLayerC()
optimizer = AdamW(model.parameters(), lr=0.00015, betas=(0.9, 0.95), )

for epoch in range(args.epochs):
    model = model.to(device)
    for idx, (batch, _) in enumerate(training_loader):
        images, mask = batch
        B = images.shape[0]
        img = images.to(device, non_blocking=True).float()
        mask_bool = mask.to(device, non_blocking=True).to(torch.bool)
        print(images.shape)
        with torch.no_grad():
            labels = HogLayer(images)[mask_bool].reshape(B, -1, 3*nbins*(cpb**2))  # B N classes

        with torch.cuda.amp.autocast():
            outputs = model(images, mask_bool)
            loss = loss_func(input=outputs, target=labels)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)



