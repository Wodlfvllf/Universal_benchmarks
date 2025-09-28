import argparse
import subprocess
from pathlib import Path
import sys

def run_all_in_category(category: str, model_type: str, model_name: str, model_config: str):
    """
    Finds and runs all benchmarks in a given category.
    """
    # The root directory of the benchmarks
    benchmarks_root = Path(__file__).parent.parent / 'benchmarks'
    category_path = benchmarks_root / category

    if not category_path.is_dir():
        print(f"Error: Category '{category}' not found at {category_path}")
        return

    print(f"--- Running all benchmarks in category: {category} ---")

    # Find all config.yaml files in the category directory
    benchmark_configs = list(category_path.glob('**/config.yaml'))

    if not benchmark_configs:
        print(f"No benchmarks found in category '{category}'.")
        return

    for config_file in benchmark_configs:
        # The benchmark name is the path relative to the benchmarks_root
        benchmark_name = config_file.parent.relative_to(benchmarks_root).as_posix()
        
        print(f"\n>>> Executing benchmark: {benchmark_name}\n")

        command = [
            sys.executable, # Use the same python interpreter
            str(Path(__file__).parent / 'run_benchmark.py'),
            '--benchmark', benchmark_name,
            '--model_type', model_type,
            '--model_name', model_name,
        ]
        if model_config:
            command.extend(['--model_config', model_config])

        try:
            # We run the benchmark script as a subprocess
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark {benchmark_name}: {e}")
            print("Continuing to the next benchmark...")
        except FileNotFoundError:
            print(f"Error: Could not find the 'run_benchmark.py' script.")
            break

    print(f"\n--- Finished all benchmarks in category: {category} ---")

def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks in a specific category.")
    parser.add_argument('--category', type=str, required=True, help='The category of benchmarks to run (e.g., \'llm\').')
    parser.add_argument('--model_type', type=str, required=True, default='huggingface', help='The type of the model to use.')
    parser.add_argument('--model_name', type=str, required=True, help='The name or path of the model to evaluate.')
    parser.add_argument('--model_config', type=str, help='Path to a JSON file with model configuration.')

    args = parser.parse_args()

    run_all_in_category(args.category, args.model_type, args.model_name, args.model_config)

if __name__ == "__main__":
    main()
