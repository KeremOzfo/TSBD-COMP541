import torch

""" Helper functions related to GPU/Cluster """
def select_least_used_gpu():
    """Select the CUDA device with the least used memory."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    num_devices = torch.cuda.device_count()
    if num_devices == 1:
        return torch.device('cuda:0')
    
    device_memory_usage = []
    for i in range(num_devices):
        try:
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            device_memory_usage.append((i, used))
        except Exception:
            device_memory_usage.append((i, 0))
    
    device_memory_usage.sort(key=lambda x: x[1])
    least_used_device_id = device_memory_usage[0][0]
    
    return torch.device(f'cuda:{least_used_device_id}')
