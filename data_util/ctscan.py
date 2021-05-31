"""
This is a loader for our specific purpouse, please implement your own loader.
"""
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

root_dir = '../datasets/750_90'
root_dir_mask = '../datasets/BORDERS'
root_dir_mask_cont = '../datasets/Contrast_masks/CONTRAST'


def get_sample_by_path(fixed_path, moving_path):
    fixed = ToTensor()(Image.open(os.path.join(root_dir, fixed_path)))
    moving = ToTensor()(Image.open(os.path.join(root_dir, moving_path)))

    # Add channel dim
    fixed = fixed.unsqueeze(1)
    moving = moving.unsqueeze(1)

    return fixed, moving


def get_mask_path(moving_path):
    info = moving_path.split('/')
    patient = info[0]
    slice = info[1]
    time = info[2].split('.')[0]
    mask_path = patient + '__' + slice + '_' + time + '.npy'

    return mask_path


def get_sample_by_path_contrast(fixed_path, moving_path):
    fixed = ToTensor()(Image.open(os.path.join(root_dir, fixed_path)))
    moving = ToTensor()(Image.open(os.path.join(root_dir, moving_path)))
    mask_path_f = get_mask_path(fixed_path)
    mask_path_m = get_mask_path(moving_path)
    mask_contrast_f = ToTensor()(Image.fromarray(np.load(os.path.join(root_dir_mask_cont, mask_path_f))))
    mask_contrast_m = ToTensor()(Image.fromarray(np.load(os.path.join(root_dir_mask_cont, mask_path_m))))

    # Add channel dim
    fixed = fixed.unsqueeze(1)
    moving = moving.unsqueeze(1)
    mask_contrast_f = mask_contrast_f.unsqueeze(1)
    mask_contrast_m = mask_contrast_m.unsqueeze(1)

    return fixed, moving, mask_contrast_f, mask_contrast_m


def get_sample_by_path_borders(fixed_path, moving_path):
    fixed = ToTensor()(Image.open(os.path.join(root_dir, fixed_path)))
    moving = ToTensor()(Image.open(os.path.join(root_dir, moving_path)))
    mask_path = get_mask_path(moving_path)
    mask_border = ToTensor()(Image.fromarray(np.load(os.path.join(root_dir_mask, mask_path))))
    # Add channel dim
    fixed = fixed.unsqueeze(1)
    moving = moving.unsqueeze(1)
    mask_border = mask_border.unsqueeze(1)

    return fixed, moving, mask_border


def get_batch(fixed_path_list, moving_batch_list):
    fixed_images = []
    moving_images = []
    for f, m in zip(fixed_path_list, moving_batch_list):
        print(f"\t\tFixed:{f}, Moving: {m}")
        fixed, moving = get_sample_by_path(f, m)
        fixed_images.append(fixed)
        moving_images.append(moving)

    if len(fixed_path_list) == 1:
        return fixed_images[0], moving_images[0]

    fixed_batch = torch.cat(fixed_images, dim=0)
    moving_batch = torch.cat(moving_images, dim=0)

    return fixed_batch, moving_batch


def get_batch_borders(fixed_path_list, moving_batch_list):
    fixed_images = []
    moving_images = []
    masks_borders = []
    for f, m in zip(fixed_path_list, moving_batch_list):
        print(f"\t\tFixed:{f}, Moving: {m}")
        fixed, moving, mask = get_sample_by_path_borders(f, m)
        fixed_images.append(fixed)
        moving_images.append(moving)
        masks_borders.append(mask)

    if len(fixed_path_list) == 1:
        return fixed_images[0], moving_images[0], masks_borders[0]

    fixed_batch = torch.cat(fixed_images, dim=0)
    moving_batch = torch.cat(moving_images, dim=0)
    mask_batch = torch.cat(masks_borders, dim=0)

    return fixed_batch, moving_batch, mask_batch


def get_batch_contrast(fixed_path_list, moving_batch_list):
    fixed_images = []
    moving_images = []
    masks_f = []
    masks_m = []
    for f, m in zip(fixed_path_list, moving_batch_list):
        print(f"\t\tFixed:{f}, Moving: {m}")
        fixed, moving, mask_f_, mask_m_ = get_sample_by_path_contrast(f, m)
        fixed_images.append(fixed)
        moving_images.append(moving)
        masks_f.append(mask_f_)
        masks_m.append(mask_m_)

    if len(fixed_path_list) == 1:
        return fixed_images[0], moving_images[0], masks_f[0], masks_m[0]

    fixed_batch = torch.cat(fixed_images, dim=0)
    moving_batch = torch.cat(moving_images, dim=0)
    mask_f_batch = torch.cat(masks_f, dim=0)
    mask_m_batch = torch.cat(masks_m, dim=0)

    return fixed_batch, moving_batch, mask_f_batch, mask_m_batch


def sample_generator(filename, batch_size=1):
    index = 0
    file_size = len(open(filename, 'r').readlines())

    while True:
        fsamples = open(filename, 'r')
        fixed_list = []
        moving_list = []
        for n, line in enumerate(fsamples):
            if n < index:
                continue
            if n < index + batch_size:
                split = line.split(",")
                fixed_list.append(split[0].strip())
                moving_list.append(split[1].strip())
            else:
                break
        if len(fixed_list) == 0 or len(moving_list) == 0:
            print("Empty List")

        if index + batch_size >= file_size:
            index = 0
        else:
            index = index + batch_size

        yield get_batch(fixed_list, moving_list)


def sample_generator_borders(filename, batch_size=1):
    index = 0
    file_size = len(open(filename, 'r').readlines())

    while True:
        fsamples = open(filename, 'r')
        fixed_list = []
        moving_list = []
        for n, line in enumerate(fsamples):
            if n < index:
                continue
            if n < index + batch_size:
                split = line.split(",")
                fixed_list.append(split[0].strip())
                moving_list.append(split[1].strip())
            else:
                break
        if len(fixed_list) == 0 or len(moving_list) == 0:
            print("Empty List")

        if index + batch_size >= file_size:
            index = 0
        else:
            index = index + batch_size

        yield get_batch_borders(fixed_list, moving_list)


def sample_generator_contrast(filename, batch_size=1):
    index = 0
    file_size = len(open(filename, 'r').readlines())

    while True:
        fsamples = open(filename, 'r')
        fixed_list = []
        moving_list = []
        for n, line in enumerate(fsamples):
            if n < index:
                continue
            if n < index + batch_size:
                split = line.split(",")
                fixed_list.append(split[0].strip())
                moving_list.append(split[1].strip())
            else:
                break
        if len(fixed_list) == 0 or len(moving_list) == 0:
            print("Empty List")

        if index + batch_size >= file_size:
            index = 0
        else:
            index = index + batch_size

        yield get_batch_contrast(fixed_list, moving_list)


if __name__ == '__main__':
    generator = iter(sample_generator_contrast('../train.txt', batch_size=10))

    for i in range(3):
        fixed, moving, mask = next(generator)
        print(fixed.size())
        print(moving.size())
        print(mask.size())


