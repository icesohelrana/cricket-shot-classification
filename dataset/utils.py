import torch
import re
def batching_frame_seq(data):
    video_tensor = [it[0] for it in data]
    class_label_tensor = [it[1] for it in data]
    video_info = [it[2] for it in data]
    max_frame_len = int(max(video_info))
    for i,a_tensor in enumerate(video_tensor):
        zeros = torch.zeros(max_frame_len,a_tensor.shape[1],a_tensor.shape[2],a_tensor.shape[3])
        zeros[:a_tensor.shape[0],:,:,:]=a_tensor
        video_tensor[i]=zeros
    return torch.stack(video_tensor),torch.stack(class_label_tensor),torch.tensor(video_info)
def get_numeric_sort_key(filename):
  numbers = re.findall(r'\d+', filename)
  if numbers:
    return int(''.join(numbers))
  else:
    # No numbers found, return 0 for sorting
    return 0