import argparse
import importlib
import pkgutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.registry import BenchmarkRegistry

def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, help='Benchmark name')
    parser.add_argument('--model', required=True, help='Model identifier')
    parser.add_argument('--model-config', type=str, help='Model config file')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    
    args = parser.parse_args()

    # Dynamically import and register benchmarks
    import_submodules('benchmarks')

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