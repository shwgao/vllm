# compare the results of DTP and TP
# For DTP, use data_parallel_size=1, tensor_parallel_size=2 and enable DTP
# For TP, use data_parallel_size=1, tensor_parallel_size=1 and disable DTP
# Then compare the results of DTP and TP are the same

import argparse
from vllm import LLM, SamplingParams
import vllm.distributed as dist

def run_comparison():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Compare DTP and TP results.")
    parser.add_argument("--model", type=str, default="./models/gradientai-Llama-3-8B-Instruct-Gradient-1048k", help="Model path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Test prompt")
    args = parser.parse_args()

    # Test prompt and sampling parameters
    prompt = args.prompt
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

    # Test DTP configuration
    print("Testing DTP configuration...")
    dist.set_dtp_group_state(True)
    dist.add_more_parallel_groups(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    
    # Initialize LLM with DTP settings
    dtp_llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        # Note: DTP is controlled by the distributed state, not LLM parameters
    )
    
    dtp_outputs = dtp_llm.generate([prompt], sampling_params)
    dtp_result = dtp_outputs[0].outputs[0].text
    print(f"DTP result: {dtp_result}")

    # Test TP configuration
    print("Testing TP configuration...")
    dist.set_dtp_group_state(False)
    dist.add_more_parallel_groups(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    
    # Initialize LLM with TP settings
    tp_llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
    )
    
    tp_outputs = tp_llm.generate([prompt], sampling_params)
    tp_result = tp_outputs[0].outputs[0].text
    print(f"TP result: {tp_result}")

    # Compare results
    print("\nComparison:")
    if dtp_result.strip() == tp_result.strip():
        print("✓ DTP and TP results are the same.")
    else:
        print("✗ DTP and TP results differ.")
        print(f"DTP: '{dtp_result}'")
        print(f"TP:  '{tp_result}'")

if __name__ == "__main__":
    run_comparison()