import argparse
from benchmarks.registry import BenchmarkRegistry
from benchmarks.llm.glue.runner import GlueBenchmarkRunner
from benchmarks.multilingual.wmt23.runner import WMT23BenchmarkRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True, help='Benchmark name')
    parser.add_argument('--model', required=True, help='Model identifier')
    parser.add_argument('--model-config', type=str, help='Model config file')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    
    args = parser.parse_args()

    BenchmarkRegistry.register('glue', 'llm', GlueBenchmarkRunner)
    BenchmarkRegistry.register('wmt23', 'multilingual', WMT23BenchmarkRunner)
    
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