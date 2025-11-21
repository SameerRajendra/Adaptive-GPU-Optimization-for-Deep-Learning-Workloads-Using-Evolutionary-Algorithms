import torch
import torch.cuda.nvtx as nvtx
from contextlib import contextmanager

@contextmanager
def nvtx_range(msg, color="blue"):
    """
    Wraps a code region with NVTX markers for Nsight Systems.
    This allows you to see 'Attention' or 'MLP' as distinct blocks on the GPU timeline.
    """
    # Color mapping could be expanded, simple implementation for now
    nvtx.range_push(msg)
    try:
        yield
    finally:
        nvtx.range_pop()

class WhiteBoxProfiler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def capture_step(self, input_ids):
        """
        Executes a single step with NVTX annotation.
        """
        with nvtx_range("Inference_Step"):
            return self.model(input_ids)