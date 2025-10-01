import argparse
import importlib
import pkgutil
from benchmarks.registry import BenchmarkRegistry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, help='Benchmark name')
    parser.add_argument('--model', required=True, help='Model identifier')
    parser.add_argument('--model-config', type=str, help='Model config file')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    
    args = parser.parse_args()

    # Dynamically import and register benchmarks
    for _, name, _ in pkgutil.walk_packages(['benchmarks']):
        importlib.import_module(f'benchmarks.{name}')

    # Get benchmark runner
    runner_class = BenchmarkRegistry.get_benchmark(args.benchmark)
    
    # Initialize and run
    runner = runner_class(
        model_name=args.model,
        model_config=args.model_config
    )
    
    results = runner.run()
    print(f"Benchmark complete. Results saved to {args.output_dir}")
    
if __name__ == "__main__":
    main()