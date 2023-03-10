#!/usr/bin/python3
from test import get_parser

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from utils.transforms_utils import get_imagenet_mean_std, normalize_img, pad_to_crop_sz, resize_by_scaled_short_side
from torchvision import transforms
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 
from utils.segformer import get_configured_segformer
import torch
import numpy as np
import torch.nn as nn
from datasets import load_metric


def val_transform(image, mask, base_size, scale):
    mean, std = get_imagenet_mean_std()
    # 1. shorter edge is resized
    image = resize_by_scaled_short_side(image, base_size, scale)
    
    # 2. padding (image only)
    orig_h, orig_w = image.shape[:2]  # hxwx3 or hxw
    crop_h = (np.ceil((orig_h - 1) / 32) * 32).astype(np.int32)
    crop_w = (np.ceil((orig_w - 1) / 32) * 32).astype(np.int32)
    image, _, _ = pad_to_crop_sz(image, crop_h, crop_w, mean)
    # 3
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    mask = torch.from_numpy(mask)
    # 4
    normalize_img(image, mean, std)

    return image, mask


blur_aug = transforms.GaussianBlur(kernel_size=3)
jitter_aug = transforms.ColorJitter(
    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


def train_transform(image, mask, base_size, scale):    
    # shorter edge is resized
    image = resize_by_scaled_short_side(image, base_size, scale)
    mask = resize_by_scaled_short_side(mask, base_size, scale)
    
    # to Tensor
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    mask = torch.from_numpy(mask)

    # normalize
    mean, std = get_imagenet_mean_std()
    normalize_img(image, mean, std)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(640, 640))              # for Segformer
    image = transforms.functional.crop(image, i, j, h, w)
    mask = transforms.functional.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    
    # Random vertical flipping
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
    
    image = jitter_aug(image)
    image = blur_aug(image)
    return image, mask

def get_shapes(h, w, crop_h, crop_w, base_size, scale):
    short_size = round(scale * base_size)
    orig_h = short_size
    orig_w = short_size
    # Preserve the aspect ratio
    if h > w:
        orig_h = round(short_size / float(w) * h)
    else:
        orig_w = round(short_size / float(h) * w)
    
    pad_h = max(crop_h - orig_h, 0)
    pad_w = max(crop_w - orig_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    return orig_h, orig_w, pad_h_half, pad_w_half


class CmpDataset(Dataset):
    def __init__(self, root_dir, set_type, base_size=720, scale=1, transform=None):
        """ Dataset of ...

        Parameters
        ----------
        """
        self.root_dir = root_dir
        self.base_size = base_size
        self.scale = scale
        self.transform = transform

        self.num_classes = 13
        self.color_map = {      
            (  0,   0,   0): 0,
            (  0,   0, 170): 1,
            (  0,   0, 255): 2,
            (  0,  85, 255): 3,
            (  0, 170, 255): 4,
            (  0, 255, 255): 5,
            ( 85, 255, 170): 6,
            (170, 255,  85): 7,
            (255, 255,   0): 8,
            (255, 170,   0): 9,
            (255,  85,   0): 10,
            (255,   0,   0): 11,
            (170,   0,   0): 12
        }   # np.array(img.getpalette()).reshape(256,3)[:13, :]
        self.id2label = {
            0: "unknown", 
            1: "background", 
            2: "facade",
            3: "window",
            4: "door",
            5: "cornice",
            6: "sill",
            7: "balcony", 
            8: "blind", 
            9: "deco", 
            10: "molding", 
            11: "pillar", 
            12: "shop"
        }

        self.img_paths = []
        self.mask_paths = []
        for file in sorted(os.listdir(self.root_dir)):
            if file.endswith(".jpg"):
                self.img_paths.append(os.path.join(root_dir, file))
            elif file.endswith(".png"):
                self.mask_paths.append(os.path.join(root_dir, file))
        
        # split val and test: 50/50
        idx = int(len(self.img_paths) * 0.5)
        if set_type == 'val':
            del self.img_paths[idx:]
            del self.mask_paths[idx:]
        elif set_type == 'test':
            del self.img_paths[:idx]
            del self.mask_paths[:idx]

    def __len__(self):
        """Returns dataset length."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Gets dataset item."""
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB").copy())
        mask = np.array(Image.open(mask_path).convert('P').copy())
        if self.transform:
            image, mask = self.transform(image, mask, self.base_size, self.scale)
        return {"image": image, "mask": mask}


class CmpDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, shuffle):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.train_dataset = CmpDataset(
            'data/CMP_facade_DB_base/base', 
            set_type='train',
            transform=train_transform)
        self.val_dataset = CmpDataset(
            'data/CMP_facade_DB_extended/extended', 
            set_type='val',
            transform=val_transform)
        self.test_dataset = CmpDataset(
            'data/CMP_facade_DB_extended/extended', 
            set_type='test',
            transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False
                          )


class SsiwNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        # load pretrained net for fine-tuning
        self.model = get_configured_segformer(args.num_model_classes,
                                              criterion=None, 
                                              load_imagenet_model=False)
        self.model = torch.nn.DataParallel(self.model, 
                                           device_ids=args.test_gpu, 
                                           output_device=args.test_gpu[0]) #.cuda()
        ckpt_path = args.model_path
        checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
        ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
        self.model.load_state_dict(ckpt_filter, strict=False)
        self.model.module.segmodel.auxi_net.cls_seg = nn.Conv2d(256,                                                          
                                                                args.num_model_classes,                                                     
                                                                kernel_size=(1, 1),                                                     
                                                                stride=(1, 1))

        # load embeddings list (of text labels)
        text_embs = torch.tensor(np.load(args.emb_path)).float()
        self.text_embs = text_embs / text_embs.norm(dim=1, keepdim=True)    # normalized
        self.text_embs = self.text_embs.t() # You maybe need to remove the .t() based on the shape of your saved .npy

        # metrics
        self.criterion = nn.CrossEntropyLoss()
        # self.train_mean_iou = load_metric("mean_iou")
        # self.val_mean_iou = load_metric("mean_iou")
        # self.test_mean_iou = load_metric("mean_iou")
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def configure_optimizers(self):
        ft_params = [param for name, param in self.model.named_parameters() if 'cls_seg' in name]
        optimizer = torch.optim.AdamW(ft_params, 
                                      lr=0.00006, 
                                      weight_decay=0.01)
        return optimizer
    
    def _shared_step(self, batch):
        logits = []
        masks = batch["mask"]
        masks = masks.long()
        images = batch["image"] 
        batch_sz, _, _, _ = images.shape
        image_embs, _, _ = self(images)
        for image_idx in range(batch_sz):
            image_emb = image_embs[image_idx,...]
            image_emb = image_emb.unsqueeze(0)
            image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
            image_emb = image_emb.permute(0, 2, 3, 1) @ self.text_embs.cuda()  # [1, H, W, num_cls]
            image_emb = image_emb.permute(0, 3, 1, 2)  # [1, num_cls, H, W]
            logits.append(image_emb)
        if len(logits) == 1:
            logits = logits[0]
        else:
            logits = torch.cat(logits, dim=0)
        return logits, masks
    
    def training_step(self, batch, batch_idx):
        logits, masks = self._shared_step(batch)
        loss = self.criterion(logits, masks)
        self.log("train_CE", loss, on_step=False, on_epoch=True,             
                 prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):        
        logits, masks = self._shared_step(batch)
        _, _, crop_h, crop_w = logits.shape
        _, h, w = masks.shape

        base_size = self.trainer.val_dataloaders[0].dataset.base_size
        scale = self.trainer.val_dataloaders[0].dataset.scale
        orig_h, orig_w, pad_h_half, pad_w_half = get_shapes(
            h, w, crop_h, crop_w, base_size, scale
        )
        logits = logits[:, :, pad_h_half:pad_h_half + orig_h, pad_w_half:pad_w_half + orig_w]
        logits = nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        
        loss = self.criterion(logits, masks)
        self.log("val_CE", loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True)
        
        # preds = logits.argmax(dim=1)
        # self.train_mean_iou.add_batch(
        #     predictions=preds.detach().cpu().numpy(), 
        #     references=masks.detach().cpu().numpy()
        # )
        # if batch_idx % self.metrics_interval == 0:

        #     metrics = self.train_mean_iou.compute(
        #         num_labels=self.num_classes, 
        #         ignore_index=255, 
        #         reduce_labels=False,
        #     )
            
        #     metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
        #     for k,v in metrics.items():
        #         self.log(k,v)
            
        #     return(metrics)
        # else:
        #     return({'loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        logits, masks = self._shared_step(batch)
        _, _, crop_h, crop_w = logits.shape
        _, h, w = masks.shape

        base_size = self.trainer.test_dataloaders[0].dataset.base_size
        scale = self.trainer.test_dataloaders[0].dataset.scale
        orig_h, orig_w, pad_h_half, pad_w_half = get_shapes(
            h, w, crop_h, crop_w, base_size, scale
        )
        logits = logits[:, :, pad_h_half:pad_h_half + orig_h, pad_w_half:pad_w_half + orig_w]
        logits = transforms.functional.resize(logits, (h, w)) # bilinear

        loss = self.criterion(logits, masks)
        return loss


def main(args):
    datamodule = CmpDatamodule(batch_size=1, num_workers=4, shuffle=True)
    loss_checkpoint = ModelCheckpoint(monitor="val_CE", 
                                      mode="min",
                                      save_top_k=2,
                                      verbose=True,
                                      filename="{epoch}_{val_loss:.3f}")
    
    # exp_name = f"{args.criterion}_{args.ft_type}_{args.gray}_"
    # exp_name += f"{args.batch_size}_{args.lr}_{args.version}_{args.jitter}_{args.crop_pad}"
    exp_name = "1st_exp"
    tb_logger = pl.loggers.TensorBoardLogger("experiments/", name=exp_name)
    
    model = SsiwNet(args)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=args.test_gpu,                                            
                         logger=tb_logger,                                            
                         callbacks=[loss_checkpoint],                                            
                         max_epochs=2)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    args = get_parser()
    main(args)
