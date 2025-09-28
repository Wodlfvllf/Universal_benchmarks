import argparse
import json
from pathlib import Path
import pandas as pd

def analyze_results(results_dir: str):
    """
    Analyzes all benchmark results in a directory and prints a summary table.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"Error: Directory not found: {results_dir}")
        return

    all_results = []

    for result_file in results_path.glob('*.json'):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Extract info from filename and content
            # Filename format: {benchmark_name}_{model_name_slug}_{timestamp}.json
            parts = result_file.name.split('_')
            benchmark_name = parts[0]
            model_name = "_".join(parts[1:-1]) # Handle model names with slashes

            # Find the aggregate results
            if 'aggregate' in data:
                agg_metrics = data['aggregate']
                flat_metrics = {
                    'Benchmark': benchmark_name,
                    'Model': model_name,
                }
                # Flatten the metrics dictionary
                for metric, value in agg_metrics.items():
                    flat_metrics[metric] = f"{value:.4f}"
                all_results.append(flat_metrics)

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Could not process file {result_file}: {e}")

    if not all_results:
        print("No valid, aggregated results found to display.")
        return

    # Create a pandas DataFrame for pretty printing
    df = pd.DataFrame(all_results)
    df = df.set_index(['Benchmark', 'Model'])
    
    print("\n--- Benchmark Results Summary ---")
    print(df.to_string())

def main():
    parser = argparse.ArgumentParser(description="Analyze and summarize benchmark results.")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/raw',
        help='Directory containing the raw JSON result files.'
    )
    args = parser.parse_args()

    analyze_results(args.results_dir)

if __name__ == "__main__":
    main()
