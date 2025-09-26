import platform
import time
import torch

# ---- Backend selection (Windows uses DirectML if available; Linux/mac use CUDA if available; else CPU) ----
def get_device():
    system = platform.system().lower()

    # Prefer DirectML on Windows if available
    if system == "windows":
        try:
            import torch_directml  # type: ignore
            dml = torch_directml.device()
            print("DirectML available. Using DirectML backend.")
            # Note: this will show up as 'privateuseone' in torch.device/type fields.
            return dml
        except Exception as e:
            print(f"DirectML not available ({e}). Falling back to CUDA/CPU.")

    # Non-Windows (or DML missing): try CUDA, then CPU
    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA.")
        return torch.device("cuda")

    print("Using CPU.")
    return torch.device("cpu")


device = get_device()
print('Using device:', device)

# ---- Device-specific info ----
# CUDA-specific diagnostics
if isinstance(device, torch.device) and device.type == 'cuda':
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('  Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('  Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
else:
    # For DirectML, torch sees device.type as 'privateuseone'
    dtype = device.type if isinstance(device, torch.device) else type(device).__name__
    print(f"{dtype} device is being used.")
    # We can't reliably query GPU name via torch_directml; keep output generic.

# Extra CUDA details block (only when CUDA available)
if torch.cuda.is_available():
    print("\nCUDA is available. Here are some details:")
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU Device Index:", torch.cuda.current_device())
    print("Current GPU Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# ---- Workload (matrix multiply) ----
# NOTE: Creating gigantic tensors directly on non-standard devices (like DirectML) via factory kwargs
# can be flaky; safest path is create on CPU then .to(device).
def allocate_tensors(size, device):
    try:
        x = torch.randn(size, size)  # CPU first
        y = torch.randn(size, size)
        x = x.to(device)
        y = y.to(device)
        return x, y
    except RuntimeError as e:
        # Simple OOM backoff for huge sizes
        if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
            fallback = max(1024, size // 4)
            print(f"OOM at size {size}x{size}. Retrying with {fallback}x{fallback}...")
            x = torch.randn(fallback, fallback).to(device)
            y = torch.randn(fallback, fallback).to(device)
            return x, y
        raise

size = 50000  # your original value; may downshift automatically if OOM
x, y = allocate_tensors(size, device)

# Synchronize for accurate timing (CUDA only)
if isinstance(device, torch.device) and device.type == 'cuda':
    torch.cuda.synchronize()

start_time = time.time()
result = torch.matmul(x, y)
if isinstance(device, torch.device) and device.type == 'cuda':
    torch.cuda.synchronize()
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
print("Result Tensor Device Type:", result.device.type if hasattr(result.device, 'type') else str(result.device))
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

# Some storage APIs can vary across PyTorch versions; guard them.
try:
    print("Result Tensor Storage Size (in bytes):", result.storage().size() * result.element_size())
    print(f"Result Tensor Storage Offset: {result.storage_offset()}")
except Exception as e:
    print(f"Storage info not available: {e}")

# These private/autocast fields may not exist on all builds/backends; guard them.
for label, getter in [
    ("Has Autograd Metadata", lambda: result._has_autocast_metadata()),
    ("Autocast Cache Enabled", lambda: result._autocast_cache_enabled()),
]:
    try:
        print(f"Result Tensor {label}: {getter()}")
    except Exception as e:
        print(f"Result Tensor {label}: N/A ({e})")

print(f"Result Tensor Device Index: {result.get_device() if result.is_cuda else 'N/A'}")
print(f"Result Tensor is Shared: {result.is_shared()}")
print(f"Result Tensor is Pinned: {result.is_pinned()}")
print(f"Result Tensor is Nested: {result.is_nested()}")

if result.is_nested():
    # Nested tensor attributes are version-dependent; guard each call.
    def safe_call(name, fn):
        try:
            return fn()
        except Exception as e:
            return f"N/A ({e})"
    print(f"Result Tensor Nested Size: {safe_call('size', lambda: result.get_nested_size())}")
    print(f"Result Tensor Nested Dim: {safe_call('dim', lambda: result.get_nested_dim())}")
    print(f"Result Tensor Nested Stride: {safe_call('stride', lambda: result.get_nested_stride())}")
    print(f"Result Tensor Nested Offset: {safe_call('offset', lambda: result.get_nested_offset())}")
    print(f"Result Tensor Nested Layout: {safe_call('layout', lambda: result.get_nested_layout())}")
    print(f"Result Tensor Nested Device: {safe_call('device', lambda: result.get_nested_device())}")
    print(f"Result Tensor Nested Requires Grad: {safe_call('rg', lambda: result.get_nested_requires_grad())}")
    print(f"Result Tensor Nested Is Leaf: {safe_call('leaf', lambda: result.get_nested_is_leaf())}")
    print(f"Result Tensor Nested Is Contiguous: {safe_call('contig', lambda: result.get_nested_is_contiguous())}")
    print(f"Result Tensor Nested Numel: {safe_call('numel', lambda: result.get_nested_numel())}")
    print(f"Result Tensor Nested Element Size: {safe_call('elem', lambda: result.get_nested_element_size())}")
    print(f"Result Tensor Nested Storage Size: {safe_call('stor', lambda: result.get_nested_storage_size())}")
    print(f"Result Tensor Nested Storage Offset: {safe_call('storoff', lambda: result.get_nested_storage_offset())}")

print("Time taken for multiplication: {:.2f} seconds".format(end_time - start_time))
print("Done")
