import argparse
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.registry import BenchmarkRegistry

def main():
    parser = argparse.ArgumentParser(description="Run a benchmark from the Universal Model Benchmark suite.")
    parser.add_argument('--benchmark', type=str, required=True, help='The name of the benchmark to run (e.g., \'llm/glue\').')
    parser.add_argument('--model_type', type=str, required=True, default='huggingface', help='The type of the model to use (e.g., \'huggingface\').')
    parser.add_argument('--model_name', type=str, required=True, help='The name or path of the model to evaluate (e.g., \'gpt2\').')
    parser.add_argument('--model_config', type=str, help='Path to a JSON file with model configuration.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the results.')

    args = parser.parse_args()

    print(f"--- Running Benchmark: {args.benchmark} ---")
    print(f"Model: {args.model_name} (Type: {args.model_type})")

    # Load model config if provided
    model_config = {}
    if args.model_config:
        import json
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    
    # Add output_dir to the config so the runner can use it
    model_config['output_dir'] = args.output_dir

    try:
        # Get the benchmark runner class from the registry
        runner_class = BenchmarkRegistry.get_benchmark_runner(args.benchmark)

        # Instantiate and run the benchmark
        runner = runner_class(
            benchmark_name=args.benchmark,
            model_type=args.model_type,
            model_name=args.model_name,
            model_config=model_config
        )
        
        results = runner.run()
        print("--- Benchmark Complete ---")
        # The runner saves the results, but we can print a summary here if desired
        if 'aggregate' in results:
            print("Aggregate Results:", results['aggregate'])

    except Exception as e:
        print(f"An error occurred during the benchmark run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
