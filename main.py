import torch

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('  Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('  Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
else:
    print(device.type, "device is being used.")

if torch.cuda.is_available():
    print("\nCUDA is available. Here are some details:")
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU Device Index:", torch.cuda.current_device())
    print("Current GPU Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Perform a simple matrix multiplication on the selected device
size = 50000
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)

# Synchronize CUDA operations for accurate timing if on GPU
if device.type == 'cuda': torch.cuda.synchronize()

import time
start_time = time.time()

result = torch.matmul(x, y)

if device.type == 'cuda': torch.cuda.synchronize()

end_time = time.time()

print("Element 1:")
print(x)
print("Element 2:")
print(y)
print("Result:")
print(result)
print(f"\nMatrix multiplication on {device} took {end_time - start_time:.4f} seconds.")

print("Multiplication Result Shape:", result.shape)
print("Result Tensor Device:", result.device)
print("Result Tensor Device Type:", result.device.type)
print("Result Tensor Type:", result.dtype)
print("Result Tensor Requires Grad:", result.requires_grad)
print("Result Tensor Memory Size (in bytes):", result.element_size() * result.nelement())
print("Result Tensor Mean Value:", result.mean().item())
print("Result Tensor Std Dev:", result.std().item())
print("Result Tensor Min Value:", result.min().item())
print("Result Tensor Max Value:", result.max().item())
print("Result Tensor Sum:", result.sum().item())
print("Result Tensor Variance:", result.var().item())
print("Result Tensor is Contiguous:", result.is_contiguous())
print("Result Tensor Numel (Total Elements):", result.numel())
print("Result Tensor Stride:", result.stride())
print("Result Tensor Layout:", result.layout)
print("Result Tensor is Leaf:", result.is_leaf)
print("Result Tensor is Non-Blocking:", result.is_nonblocking)
print(f"Result Tensor is Sparse: {result.is_sparse}")
print(f"Result Tensor is Quantized: {result.is_quantized}")
print("Result Tensor Storage Size (in bytes):", result.storage().size() * result.element_size())
print(f"Result Tensor Storage Offset: {result.storage_offset()}")
print(f"Result Tensor Has Autograd Metadata: {result._has_autocast_metadata()}")
print(f"Result Tensor Autocast Cache Enabled: {result._autocast_cache_enabled()}")
print(f"Result Tensor Device Index: {result.get_device() if result.is_cuda else 'N/A'}")
print(f"Result Tensor is Shared: {result.is_shared()}")
print(f"Result Tensor is Pinned: {result.is_pinned()}")
print(f"Result Tensor is Nested: {result.is_nested()}")
print(f"Result Tensor Nested Size: {result.get_nested_size() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Dim: {result.get_nested_dim() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Stride: {result.get_nested_stride() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Offset: {result.get_nested_offset() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Layout: {result.get_nested_layout() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Device: {result.get_nested_device() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Requires Grad: {result.get_nested_requires_grad() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Is Leaf: {result.get_nested_is_leaf() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Is Contiguous: {result.get_nested_is_contiguous() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Numel: {result.get_nested_numel() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Element Size: {result.get_nested_element_size() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Storage Size: {result.get_nested_storage_size() if result.is_nested() else 'N/A'}")
print(f"Result Tensor Nested Storage Offset: {result.get_nested_storage_offset() if result.is_nested() else 'N/A'}")
print("Time taken for multiplication: {:.2f} seconds".format(end_time - start_time))
print("Done")
