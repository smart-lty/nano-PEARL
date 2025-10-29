import sys
import copy
import argparse
import os
from random import seed
import time
import torch
from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger

def parse_args():
    parser = argparse.ArgumentParser(description='PEARL Benchmark Tool')
    
    parser.add_argument('--draft-model', '-d', type=str, required=True,
                       help='Draft model path (required)')
    parser.add_argument('--target-model', '-t', type=str, required=True,
                       help='Target model path (required)')
    parser.add_argument('--draft-tp', type=int, default=1,
                       help='Draft model tensor parallel size (default: 1)')
    parser.add_argument('--target-tp', type=int, default=2,
                       help='Target model tensor parallel size (default: 2)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization (default: 0.9)')
    parser.add_argument('--temperature', '-temp', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate (default: 200)')
    parser.add_argument('--ignore-eos', '-noeos', action='store_true',
                       help='Ignore EOS token (default: False)')
    parser.add_argument('--run-ar-benchmark', '-ar', action='store_true',
                       help='Run AR (Autoregressive) benchmark (default: False)')
    parser.add_argument('--custom-prompts', '-p', type=str, nargs='+',
                       help='Custom prompts for benchmark')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (default: False)')
    
    return parser.parse_args()

def get_default_prompts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(script_dir, 'static', 'default_prompts.txt')
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

if __name__ == "__main__":
    args = parse_args()
    
    seed(args.seed)
    draft_model_path = args.draft_model
    target_model_path = args.target_model
    config = PEARLConfig(
        draft_model_path, 
        target_model_path, 
        draft_tensor_parallel_size=args.draft_tp, 
        target_tensor_parallel_size=args.target_tp, 
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    engine = PEARLEngine(config)

    # warmup
    prompt = "Benchmark:"
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=512)
    engine.add_request(prompt, sampling_params)
    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
    MAT = [sum(n) / len(n) for n in num_acc_tokens]
    logger.info(f"[Warmup] Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {sum(num_tokens) / elapsed_time:.2f}tok/s, MAT: {MAT}")

    prompts = args.custom_prompts if args.custom_prompts else get_default_prompts()
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        ignore_eos=args.ignore_eos, 
        max_tokens=args.max_tokens
    )
    for prompt in prompts:
        engine.add_request(prompt, copy.deepcopy(sampling_params))

    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.bench_generate(num_pearl_steps=100)
    MAT = [sum(n) / len(n) for n in num_acc_tokens]

    if args.verbose:
        for prompt, output_text in zip(prompts, output_text):
            logger.info(f"Prompt: \n{prompt}", color="yellow")
            logger.info(f"Completion: \n{output_text}")

    PEARL_throughput = sum(num_tokens) / elapsed_time
    logger.info(f"num_tokens: {num_tokens}, MAT: {MAT}")
    logger.info(f"[PEARL Generate] Batch Size: {len(prompts)} Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {sum(num_tokens) / elapsed_time:.2f}tok/s, MAT: {MAT}")

    if args.run_ar_benchmark:
        sampling_params = SamplingParams(
            temperature=args.temperature, 
            ignore_eos=args.ignore_eos, 
            max_tokens=args.max_tokens
        )
        for prompt in prompts:
            engine.add_request(prompt, copy.deepcopy(sampling_params))

        output_text, num_tokens, _, elapsed_time = engine.AR_generate()
        if args.verbose:
            for prompt, output_text in zip(prompts, output_text):
                logger.info(f"Prompt: \n{prompt}", color="yellow")
                logger.info(f"Completion: \n{output_text}")
        AR_throughput = sum(num_tokens) / elapsed_time
        logger.info(f"[AR Generate] Batch Size: {len(prompts)} Total: {sum(num_tokens)}tok, Time: {elapsed_time:.2f}s, Throughput: {sum(num_tokens) / elapsed_time:.2f}tok/s")
        logger.info(f"PEARL Speedup: {PEARL_throughput / AR_throughput:.2f}x")