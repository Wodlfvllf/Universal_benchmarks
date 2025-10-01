import argparse
import importlib
import pkgutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarks.registry import BenchmarkRegistry

def import_submodules(package):
    """ Import all submodules of a module, recursively, including subpackages """
    import sys
    import os
    import importlib

    package_dir = package.replace('.', '/')
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                module_path = os.path.join(root, file)
                module_name = module_path.replace('/', '.').replace('.py', '')
                importlib.import_module(module_name)

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