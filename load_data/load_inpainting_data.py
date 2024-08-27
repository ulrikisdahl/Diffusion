from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse
import importlib
from torch.utils.data import DataLoader, Subset


IMG_SIZE = 256
BATCH_SIZE = 128



def show_tensor_image(image, display=True):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t.cpu()),  # Ensure tensor is on CPU
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 

    if display:
        plt.imshow(reverse_transforms(image))
    else:
        return reverse_transforms(image)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_inpaint_loader(batch_size=1, shuffle=False):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the inference results",
    )

    parser.add_argument(
            "--config",
            type=str,
            default="configs/data_config.yaml",
            help="Path of the model config file",
    )

    args = parser.parse_args()
    device = "cuda"
    outdir = args.outdir

    parsed_config = OmegaConf.load(args.config)

    dataset = instantiate_from_config(parsed_config["data"])
    dataset.setup()
    train_dataset = dataset.datasets["train"]
    print(f"LENGTH: {len(train_dataset)}")
    train_dataset = Subset(train_dataset, range(20000, 50000))

    dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle, drop_last=True)

    return dataloader