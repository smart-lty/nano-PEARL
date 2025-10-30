import sys
import copy
import argparse
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from random import seed
import torch
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger
import copy


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='JSONL Benchmark Evaluation Tool for nano-PEARL')
    
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
    parser.add_argument('--num-pearl-steps', type=int, default=100, 
                       help='Number of PEARL steps for bench_generate in evaluation (default: 100)')
    parser.add_argument('--ignore-eos', '-noeos', action='store_true',
                       help='Ignore EOS token (default: False)')
    
    # Evaluation arguments
    parser.add_argument('--dataset', type=str, 
                       choices=['HumanEval', 'CNNDM', 'AIME', 'GSM8K', 'all'],
                       default='all', help='Dataset to evaluate (default: all)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
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


def load_jsonl_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSONL data file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line {i+1}: {e}")
                continue
    return data


def extract_prompts(data: List[Dict[str, Any]], dataset_name: str) -> List[str]:
    """Extract prompts from data"""
    prompts = []
    for item in data:
        prompt = item.get('turns', [''])[0] if item.get('turns') else ''
        prompts.append(prompt.strip())
    return prompts


def run_benchmark(engine: PEARLEngine, prompts: List[str], sampling_params: SamplingParams, 
                 dataset_name: str, batch_size: int = 1, run_ar: bool = False, num_pearl_steps: int = 100) -> Tuple[List[str], Dict[str, float], float]:
    """Run benchmark test"""
    logger.info(f"Starting evaluation of {dataset_name} dataset, sample count: {len(prompts)}, batch size: {batch_size}")
    
    all_outputs = []
    all_num_tokens = []
    all_num_acc_tokens = []
    total_elapsed_time = 0
    
    # Calculate number of complete batches (discard incomplete last batch)
    num_complete_batches = len(prompts) // batch_size
    total_samples = num_complete_batches * batch_size
    
    logger.info(f"Processing {num_complete_batches} complete batches with {total_samples} samples (discarding {len(prompts) - total_samples} incomplete samples)")
    
    # Process in batches with progress bar
    for i in tqdm(range(0, total_samples, batch_size), desc=f"Processing {dataset_name}", unit="batch"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Add current batch requests
        for prompt in batch_prompts:
            engine.add_request(prompt, copy.deepcopy(sampling_params))
        
        # PEARL generation
        output_text, num_tokens, num_acc_tokens, elapsed_time = engine.bench_generate(num_pearl_steps=num_pearl_steps)
        
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
        for i in tqdm(range(0, total_samples, batch_size), desc=f"AR Processing {dataset_name}", unit="batch"):
            batch_prompts = prompts[i:i + batch_size]
            
            for prompt in batch_prompts:
                engine.add_request(prompt, copy.deepcopy(sampling_params))
            
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
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=16384,
        max_num_seqs=128,
        max_num_batched_tokens=16384*2,
    )
    engine = PEARLEngine(config)
    
    # Warmup (optional) - align with bench.py
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
    
    # Get data file paths
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    datasets = {
        'HumanEval': os.path.join(data_dir, 'HumanEval.jsonl'),
        'CNNDM': os.path.join(data_dir, 'CNNDM.jsonl'),
        'AIME': os.path.join(data_dir, 'AIME.jsonl'),
        'GSM8K': os.path.join(data_dir, 'GSM8K.jsonl')
    }
    
    # Determine datasets to evaluate
    if args.dataset == 'all':
        eval_datasets = list(datasets.keys())
    else:
        eval_datasets = [args.dataset]
    
    # Run evaluation
    all_results = {}
    
    for dataset_name in eval_datasets:
        if dataset_name not in datasets:
            logger.warning(f"Dataset {dataset_name} does not exist, skipping")
            continue
        
        data_file = datasets[dataset_name]
        if not os.path.exists(data_file):
            logger.warning(f"Data file {data_file} does not exist, skipping")
            continue
        
        # Load data
        logger.info(f"Loading {dataset_name} dataset...")
        data = load_jsonl_data(data_file, args.max_samples)
        prompts = extract_prompts(data, dataset_name)
        
        if not prompts:
            logger.warning(f"{dataset_name} dataset has no valid prompts, skipping")
            continue
        
        # Run benchmark test
        outputs, metrics, elapsed_time = run_benchmark(
            engine, prompts, sampling_params, dataset_name, args.bs, args.run_ar_benchmark, args.num_pearl_steps
        )
        
        # Store results
        results = {
            'dataset': dataset_name,
            'num_samples': len(prompts),
            'metrics': metrics,
            'elapsed_time': elapsed_time,
            'outputs': outputs[:5] if args.verbose else []  # Only save first 5 outputs for debugging
        }
        
        all_results[dataset_name] = results
        
        logger.info(f"{dataset_name} evaluation completed")
    
    # Print overall report to console
    if all_results:
        print("\n" + "=" * 60)
        print("nano-PEARL Benchmark Overall Report")
        print("=" * 60)
        
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name}:")
            print(f"  Sample count: {results['num_samples']}")
            print(f"  PEARL throughput: {results['metrics']['pearl_throughput']:.2f} tok/s")
            if results['metrics']['ar_throughput'] > 0:
                print(f"  AR throughput: {results['metrics']['ar_throughput']:.2f} tok/s")
                print(f"  Speedup: {results['metrics']['speedup']:.2f}x")
        
        print("\n" + "=" * 60)
        logger.info("All evaluations completed!")


if __name__ == "__main__":
    main()
