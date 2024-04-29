import torch
from torch.utils.data import Dataset

import cv2

import os


class WRSRDataset(Dataset):
  def __init__(self, HQ_path, Ref_path, keys=('HQ', 'Ref'), is_val=False):
    self._is_val = is_val
    
    self.ref_images = self._get_files(Ref_path)
    self.gt_images = self._get_files(HQ_path)

    self.hq_key = keys[0]
    self.ref_key = keys[1]

  def _get_files(self, path):
    files = sorted(os.listdir(path))
    files = [f'{path}/{file}' for file in files]
    return files

  def _prepare_data(self, image):
    image = torch.tensor(image)
    image = torch.permute(image, (2, 0, 1))
    image = image / 255.0
    
    return image

  def __len__(self):
    return len(self.gt_images)

  def __getitem__(self, index):
    ref_image = self.ref_images[index]
    gt_image = self.gt_images[index]

    ref_image = cv2.cvtColor(cv2.imread(ref_image), cv2.COLOR_BGR2RGB)
    gt_image = cv2.cvtColor(cv2.imread(gt_image), cv2.COLOR_BGR2RGB)

    ref_image = self._prepare_data(ref_image)
    gt_image = self._prepare_data(gt_image)
    
    return {
       self.hq_key: gt_image,
       self.ref_key: ref_image,
    }