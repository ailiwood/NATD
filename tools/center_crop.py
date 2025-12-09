import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def center_crop(tensor, target_tensor):
    _, _, tensor_width = tensor.size()
    _, _, target_width = target_tensor.size()
    diff = tensor_width - target_width
    if diff == 0:
        return tensor
    elif diff < 0:
        raise ValueError("tensor is smaller than target_tensor")
    else:
        crop_left = diff // 2
        crop_right = diff - crop_left
        return tensor[:, :, crop_left:tensor_width - crop_right]
