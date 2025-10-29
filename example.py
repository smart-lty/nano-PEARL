from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger

def main():
    draft_model_path = "/path/to/draft/model"
    target_model_path = "/path/to/target/model"
    
    config = PEARLConfig(draft_model_path, target_model_path, draft_tensor_parallel_size=1, target_tensor_parallel_size=1, gpu_memory_utilization=0.9)
    engine = PEARLEngine(config)
    
    prompt = "Explain quantum computing in simple terms"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=False)
    engine.add_request(prompt, sampling_params)
    
    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
    logger.info(f"Completion:", color="yellow")
    logger.info(f"{output_text[0]}")
    logger.info(f"Tokens: {num_tokens[0]}, Time: {elapsed_time:.2f}s, Throughput: {num_tokens[0] / elapsed_time:.2f} tok/s, MAT: {sum(num_acc_tokens[0]) / len(num_acc_tokens[0])}")

if __name__ == "__main__":
    main()