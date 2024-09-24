import torch

if torch.cuda.is_available():
    print("GPU khả dụng")
    print("Số lượng GPU:", torch.cuda.device_count())
    print("Tên GPU:", torch.cuda.get_device_name(0)) # Lấy tên của GPU đầu tiên
else:
    print("GPU không khả dụng")