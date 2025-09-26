import platform
from datetime import datetime
import torch

# Optional libs for memory probing on non-CUDA backends
try:
    import psutil  # for system RAM as a fallback
except Exception:
    psutil = None

# Try NVML for VRAM stats when not using CUDA (e.g., DirectML + NVIDIA on Windows)
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_READY = True
except Exception:
    _NVML_READY = False

# ---- Helpers ----
def bytes_to_human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024

def pretty_time(ts: datetime) -> str:
    # Local human-readable timestamp
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")

def compute_required_bytes(size: int, dtype=torch.float32, include_result=True) -> int:
    # Two inputs of (size x size) and optionally the result
    bpe = torch.tensor([], dtype=dtype).element_size()
    matrices = 3 if include_result else 2
    return matrices * size * size * bpe

def get_memory_capacity_and_free(device) -> tuple[int | None, int | None, str]:
    """
    Returns (total_bytes, free_bytes, source).
    If unknown, returns (None, None, 'unknown').
    """
    # CUDA: reliable
    if isinstance(device, torch.device) and device.type == 'cuda':
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            return int(total_b), int(free_b), "cuda.mem_get_info"
        except Exception:
            try:
                props = torch.cuda.get_device_properties(0)
                return int(props.total_memory), None, "cuda.get_device_properties"
            except Exception:
                return None, None, "cuda.unknown"

    # DirectML (Windows) or other "privateuseone"
    # Try NVML first (common on NVIDIA GPUs even if not using CUDA in PyTorch)
    if _NVML_READY:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # total, free available
            return int(mem.total), int(mem.free), "nvml"
        except Exception:
            pass

    # CPU fallback (system RAM). Not the same as VRAM, but still a sanity check.
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available), "psutil"
        except Exception:
            pass

    return None, None, "unknown"

# ---- Backend selection (Windows uses DirectML if available; Linux/mac use CUDA if available; else CPU) ----
def get_device():
    system = platform.system().lower()
    print(f"___Running on {system} OS___")
    print(f"=============={'=' * len(system)}======")
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
    total_b, free_b, src = get_memory_capacity_and_free(device)
    print('Memory Usage:')
    print('  Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('  Reserved: ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
    if total_b is not None:
        print(f'  VRAM Total ({src}): {bytes_to_human(total_b)}')
    if free_b is not None:
        print(f'  VRAM Free  ({src}): {bytes_to_human(free_b)}')
else:
    # For DirectML, torch sees device.type as 'privateuseone'
    dtype = device.type if isinstance(device, torch.device) else type(device).__name__
    print(f"{dtype} device is being used.")
    total_b, free_b, src = get_memory_capacity_and_free(device)
    if total_b is not None:
        print(f'  Memory Total ({src}): {bytes_to_human(total_b)}')
    if free_b is not None:
        print(f'  Memory Free  ({src}): {bytes_to_human(free_b)}')
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
def allocate_tensors(size, device, dtype=torch.float32, safety_margin=0.90):
    """
    Allocates two (size x size) tensors on the given device, with preflight checks.

    - Prints total bytes that will be allocated for inputs and estimated result.
    - If we can determine available memory and the requirement exceeds it (after safety margin),
      we raise a RuntimeError before allocation.
    - Handles OOM with a simple backoff.
    - Catches UnicodeDecodeError seen on very large sizes and re-raises with context.
    """
    try:
        needed_b = compute_required_bytes(size, dtype=dtype, include_result=True)
        print(f"[Preflight] Intended tensor dims: {size}x{size} (dtype={dtype})")
        print(f"[Preflight] Estimated total allocation (x, y, result): {bytes_to_human(needed_b)}")

        total_b, free_b, src = get_memory_capacity_and_free(device)
        if free_b is not None:
            allowed = int(free_b * safety_margin)
            if needed_b > allowed:
                raise RuntimeError(
                    f"Requested {bytes_to_human(needed_b)} exceeds ~{int(safety_margin*100)}% of free memory "
                    f"({bytes_to_human(free_b)} via {src}). Reduce 'size' or use a smaller dtype."
                )

        # Allocate on CPU first, then move to device
        x = torch.randn(size, size, dtype=dtype)
        y = torch.randn(size, size, dtype=dtype)
        x = x.to(device)
        y = y.to(device)
        # Report actual allocations (inputs only for now)
        inputs_bytes = x.element_size() * x.nelement() + y.element_size() * y.nelement()
        est_result_bytes = size * size * x.element_size()
        print(f"[Allocated] Inputs (x+y): {bytes_to_human(inputs_bytes)}; "
              f"Estimated result: {bytes_to_human(est_result_bytes)}; "
              f"Total est: {bytes_to_human(inputs_bytes + est_result_bytes)}")
        return x, y

    except UnicodeDecodeError as e:
        # Odd console/encoding issue some users see at big sizes.
        # Provide actionable guidance.
        raise RuntimeError(
            "Encountered a UnicodeDecodeError during allocation. This is likely due to console encoding "
            "on Windows when printing large outputs. Try setting the environment variable "
            "PYTHONIOENCODING=utf-8 (or run 'chcp 65001' in cmd), and/or reduce 'size'. "
            f"Original error: {e}"
        ) from e
    except RuntimeError as e:
        # Simple OOM/backoff for huge sizes
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg:
            fallback = max(1024, size // 4)
            print(f"OOM at size {size}x{size}. Retrying with {fallback}x{fallback}...")
            x = torch.randn(fallback, fallback, dtype=dtype).to(device)
            y = torch.randn(fallback, fallback, dtype=dtype).to(device)
            return x, y
        # propagate other runtime errors (including our preflight check)
        raise

size = 10000  # your original value; may downshift automatically if OOM
x, y = allocate_tensors(size, device)

# Synchronize for accurate timing (CUDA only)
if isinstance(device, torch.device) and device.type == 'cuda':
    torch.cuda.synchronize()

start_time = datetime.now()
print(f"Starting calculation at {pretty_time(start_time)}...")
result = torch.matmul(x, y)
if isinstance(device, torch.device) and device.type == 'cuda':
    torch.cuda.synchronize()
end_time = datetime.now()
print(f"Finished calculation at {pretty_time(end_time)}.")
elapsed = end_time - start_time

print("Element 1:")
print(x)
print("Element 2:")
print(y)
print("Result:")
print(result)
print(f"\nMatrix multiplication on {device} took {elapsed}")

# Move a lightweight *view* to CPU for scalar stats to avoid DML fallback warnings.
try:
    result_cpu = result.detach().to("cpu")
except Exception as _e:
    print(f"[warn] Could not move result to CPU for stats ({_e}). Falling back to device ops.")
    result_cpu = result

print("Multiplication Result Shape:", result.shape)
print("Result Tensor Device:", result.device)
print("Result Tensor Device Type:", result.device.type if hasattr(result.device, 'type') else str(result.device))
print("Result Tensor Type:", result.dtype)
print("Result Tensor Requires Grad:", result.requires_grad)
print("Result Tensor Memory Size (in bytes):", result.element_size() * result.nelement())
try:
    print("Result Tensor Mean Value:", result_cpu.mean().item())
    print("Result Tensor Std Dev:", result_cpu.std().item())
    print("Result Tensor Min Value:", result_cpu.min().item())
    print("Result Tensor Max Value:", result_cpu.max().item())
    print("Result Tensor Sum:", result_cpu.sum().item())
    print("Result Tensor Variance:", result_cpu.var().item())
except Exception as _e:
    print(f"[warn] Stats on CPU failed ({_e}). Trying on-device...")
    try:
        print("Result Tensor Mean Value:", result.mean().item())
        print("Result Tensor Std Dev:", result.std().item())
        print("Result Tensor Min Value:", result.min().item())
        print("Result Tensor Max Value:", result.max().item())
        print("Result Tensor Sum:", result.sum().item())
        print("Result Tensor Variance:", result.var().item())
    except Exception as _e2:
        print(f"[warn] Stats on device also failed ({_e2}).")

print("Result Tensor is Contiguous:", result.is_contiguous())
print("Result Tensor Numel (Total Elements):", result.numel())
print("Result Tensor Stride:", result.stride())
print("Result Tensor Layout:", result.layout)
print("Result Tensor is Leaf:", result.is_leaf)
try:
    _nonblock_attr = getattr(result, "is_nonblocking", None)
    if callable(_nonblock_attr):
        _nb = _nonblock_attr()
    else:
        _nb = _nonblock_attr if _nonblock_attr is not None else "N/A (attribute not available in this build)"
    print("Result Tensor is Non-Blocking:", _nb)
except Exception as _e:
    print(f"Result Tensor is Non-Blocking: N/A ({_e})")
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

print("Done")
