"""
Profiling utilities for identifying performance bottlenecks
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

class ModelProfiler:
    def __init__(self, model_name: str = 'gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def profile_model(self, batch_size: int = 4, seq_length: int = 128):
        """Profile model and identify hotspots"""
        input_ids = torch.randint(0, self.tokenizer.vocab_size, 
                                 (batch_size, seq_length)).cuda()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(input_ids)
        
        # Profile
        print("Profiling model...")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = self.model(input_ids)
        
        # Print results
        print("\n" + "="*80)
        print("TOP 10 CUDA TIME OPERATIONS")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10))
        
        print("\n" + "="*80)
        print("TOP 10 MEMORY OPERATIONS")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=10))
        
        # Export detailed results
        prof.export_chrome_trace("results/profile_trace.json")
        print("\nDetailed trace saved to results/profile_trace.json")
        print("View at: chrome://tracing")
        
        return prof

def main():
    profiler = ModelProfiler(model_name='gpt2')
    prof = profiler.profile_model(batch_size=8, seq_length=256)

if __name__ == '__main__':
    main()
