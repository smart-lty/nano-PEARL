#!/usr/bin/env python3
"""
Random Input Evaluation Tool for nano-PEARL

This script evaluates nano-PEARL performance on random input_ids tensors.
"""

import argparse
import time
import os
from typing import List, Dict, Any, Tuple
from random import seed
import torch
from tqdm import tqdm
import sys
import random
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Random Input Evaluation Tool for nano-PEARL')
    
    # Model path arguments
    parser.add_argument('--draft-model', '-d', type=str, required=True,
                       help='Draft model path (required)')
    parser.add_argument('--target-model', '-t', type=str, required=True,
                       help='Target model path (required)')
    
    # Model configuration arguments
    parser.add_argument('--draft-tp', type=int, default=1,
                       help='Draft model tensor parallel size (default: 1)')
    parser.add_argument('--target-tp', type=int, default=2,
                       help='Target model tensor parallel size (default: 2)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization (default: 0.9)')
    
    # Generation arguments
    parser.add_argument('--temperature', '-temp', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate (default: 200)')
    parser.add_argument('--ignore-eos', '-noeos', action='store_true',
                       help='Ignore EOS token (default: False)')
    
    # Random input arguments
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of random samples to generate (default: 100)')
    parser.add_argument('--input-len', type=int, default=1024,
                       help='Input sequence length (default: 1024)')
    parser.add_argument('--bs', type=int, default=1,
                       help='Batch size for processing (default: 1)')
    parser.add_argument('--run-ar-benchmark', '-ar', action='store_true',
                       help='Run AR (Autoregressive) benchmark (default: False)')
    parser.add_argument('--warmup-iters', type=int, default=1,
                       help='Warmup iterations before evaluation (default: 1, set 0 to disable)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (default: False)')
    
    return parser.parse_args()


def generate_random_inputs(num_samples: int, input_len: int) -> List[List[int]]:
    """Generate random input_ids tensors"""
    inputs = [[random.randint(0, 10000) for _ in range(input_len)] for _ in range(num_samples)]
    return inputs


def run_benchmark(engine: PEARLEngine, inputs: List[List[int]], sampling_params: SamplingParams, 
                 batch_size: int = 1, run_ar: bool = False) -> Tuple[List[List[int]], Dict[str, float], float]:
    """Run benchmark test with random inputs"""
    logger.info(f"Starting evaluation with {len(inputs)} random samples, batch size: {batch_size}")
    
    all_outputs = []
    all_num_tokens = []
    all_num_acc_tokens = []
    total_elapsed_time = 0
    
    # Calculate number of complete batches (discard incomplete last batch)
    num_complete_batches = len(inputs) // batch_size
    total_samples = num_complete_batches * batch_size
    
    logger.info(f"Processing {num_complete_batches} complete batches with {total_samples} samples (discarding {len(inputs) - total_samples} incomplete samples)")
    
    # Process in batches with progress bar
    for i in tqdm(range(0, total_samples, batch_size), desc="Processing random inputs", unit="batch"):
        batch_inputs = inputs[i:i + batch_size]
        
        # Add current batch requests
        for input_ids in batch_inputs:
            engine.add_request(input_ids, copy.deepcopy(sampling_params))
        
        # PEARL generation
        output_text, num_tokens, num_acc_tokens, elapsed_time = engine.bench_generate(num_pearl_steps=100)
        # Accumulate results
        all_outputs.extend(output_text)
        all_num_tokens.extend(num_tokens)
        all_num_acc_tokens.extend(num_acc_tokens)
        total_elapsed_time += elapsed_time
    
    # Calculate overall metrics
    MAT = [sum(n) / len(n) for n in all_num_acc_tokens] if all_num_acc_tokens else [0]
    global_MAT = sum(MAT) / len(MAT)
    pearl_throughput = sum(all_num_tokens) / total_elapsed_time if total_elapsed_time > 0 else 0
    
    logger.info(f"[PEARL] Total tokens: {sum(all_num_tokens)}, time: {total_elapsed_time:.2f}s, "
               f"throughput: {pearl_throughput:.2f} tok/s, global MAT: {global_MAT}")
    
    # AR generation (if requested)
    ar_throughput = 0
    if run_ar:
        logger.info("Starting AR benchmark test...")
        all_ar_outputs = []
        all_ar_num_tokens = []
        total_ar_time = 0
        
        # Process AR generation in batches with progress bar
        for i in tqdm(range(0, total_samples, batch_size), desc="AR Processing random inputs", unit="batch"):
            batch_inputs = inputs[i:i + batch_size]
            
            for input_ids in batch_inputs:
                engine.add_request(input_ids, copy.deepcopy(sampling_params))
            
            start_time = time.time()
            ar_output_text, ar_num_tokens, _, ar_elapsed_time = engine.AR_generate()
            batch_ar_time = time.time() - start_time
            
            all_ar_outputs.extend(ar_output_text)
            all_ar_num_tokens.extend(ar_num_tokens)
            total_ar_time += ar_elapsed_time
        
        ar_throughput = sum(all_ar_num_tokens) / total_ar_time if total_ar_time > 0 else 0
        
        logger.info(f"[AR] Total tokens: {sum(all_ar_num_tokens)}, time: {total_ar_time:.2f}s, "
                   f"throughput: {ar_throughput:.2f} tok/s")
        
        if ar_throughput > 0:
            speedup = pearl_throughput / ar_throughput
            logger.info(f"PEARL speedup: {speedup:.2f}x")
    
    # Calculate basic metrics
    metrics = {
        'num_samples': len(all_outputs),
        'pearl_throughput': pearl_throughput,
        'ar_throughput': ar_throughput,
        'speedup': pearl_throughput / ar_throughput if ar_throughput > 0 else 0
    }
    
    return all_outputs, metrics, total_elapsed_time


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    seed(args.seed)
    
    # Initialize PEARL engine
    config = PEARLConfig(
        args.draft_model, 
        args.target_model, 
        draft_tensor_parallel_size=args.draft_tp, 
        target_tensor_parallel_size=args.target_tp, 
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    engine = PEARLEngine(config)
    
    # Warmup (optional) - align with bench.py (use a short text warmup)
    if args.warmup_iters > 0:
        logger.info(f"Warmup start: iters={args.warmup_iters}")
        for _ in range(args.warmup_iters):
            prompt = "Benchmark:"
            sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=512)
            engine.add_request(prompt, sampling_params)
            output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
            MAT = [sum(n) / len(n) for n in num_acc_tokens]
            logger.info(f"[Warmup] Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {sum(num_tokens) / elapsed_time:.2f}tok/s, MAT: {MAT}")
        logger.info("Warmup done")
    
    # Set sampling parameters
    # ! In PEARL bench generate, the `ignore_eos` is forced to True to avoid the early finished sequences.
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        ignore_eos=args.ignore_eos, 
        max_tokens=args.max_tokens
    )
    
    # Generate random inputs
    logger.info(f"Generating {args.num_samples} random inputs with length {args.input_len}...")
    inputs = generate_random_inputs(args.num_samples, args.input_len)
    
    # Run benchmark test
    outputs, metrics, elapsed_time = run_benchmark(
        engine, inputs, sampling_params, args.bs, args.run_ar_benchmark
    )
    
    # Print results to console
    print("\n" + "=" * 60)
    print("Random Input Evaluation Results")
    print("=" * 60)
    print(f"Sample count: {metrics['num_samples']}")
    print(f"Input length: {args.input_len}")
    print(f"PEARL throughput: {metrics['pearl_throughput']:.2f} tok/s")
    if metrics['ar_throughput'] > 0:
        print(f"AR throughput: {metrics['ar_throughput']:.2f} tok/s")
        print(f"Speedup: {metrics['speedup']:.2f}x")
    print("=" * 60)
    logger.info("Random input evaluation completed!")


if __name__ == "__main__":
    main()
