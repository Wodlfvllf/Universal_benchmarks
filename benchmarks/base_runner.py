from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from datetime import datetime

# Assuming the registries are in the parent directories and are importable
from ..tasks.registry import TaskRegistry
from ..models.registry import ModelRegistry
from ..datasets.registry import DatasetRegistry

# A basic logger setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    version: str
    description: str
    subtasks: List[Dict[str, Any]]
    dataset_config: Dict[str, Any]
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            name=config['benchmark']['name'],
            version=config['benchmark']['version'],
            description=config['benchmark'].get('description', ''),
            subtasks=config['subtasks'],
            dataset_config=config['dataset'],
            evaluation_config=config.get('evaluation', {}),
            output_config=config.get('output', {})
        )

class BaseBenchmarkRunner(ABC):
    """Base class for benchmark execution"""
    
    def __init__(self, benchmark_name: str, model_type: str, model_name: str, model_config: Optional[Dict] = None):
        self.benchmark_name = benchmark_name
        # The config path will be derived from the benchmark name
        config_path = f'/root/benchmarks/universal-model-benchmarks/benchmarks/{benchmark_name}/config.yaml'
        self.config = BenchmarkConfig.from_yaml(config_path)
        self.model = ModelRegistry.get_model(model_type, model_name, model_config)
        self.results = {}
        
    def load_data(self, subtask_config: Dict) -> Any:
        """Load dataset for a subtask"""
        dataset_name = self.config.dataset_config['name']
        dataset_sub_config = subtask_config.get('dataset_config', dataset_name)
        
        # Assuming get_dataset can handle sub-configurations
        dataset = DatasetRegistry.get_dataset(dataset_name, name=dataset_sub_config)
        return dataset
        
    def run_subtask(self, subtask_config: Dict) -> Dict:
        """Execute a single subtask"""
        logger.info(f"Running subtask: {subtask_config['name']}")
        
        # Load task implementation
        task = TaskRegistry.get_task(subtask_config['task_type'])
        
        # Load dataset for the specific split required by the subtask
        dataset = self.load_data(subtask_config)
        
        # Prepare inputs
        # This part needs to be flexible based on the task
        inputs = task.prepare_inputs(
            dataset['test'], # Assuming a 'test' split for evaluation
            input_columns=subtask_config['input_columns'],
            label_column=subtask_config.get('label_column')
        )
        
        # Get predictions
        predictions = task.predict(
            self.model,
            inputs,
            batch_size=self.config.evaluation_config.get('batch_size', 32)
        )
        
        # Calculate metrics
        metrics = task.compute_metrics(predictions, inputs)
            
        return {
            'task': subtask_config['name'],
            'metrics': metrics,
            'predictions': [p.predictions for p in predictions] if self.config.output_config.get('save_raw_predictions') else None
        }
        
    def run(self) -> Dict:
        """Execute all subtasks in the benchmark"""
        logger.info(f"Starting benchmark: {self.config.name} v{self.config.version}")
        
        for subtask_config in self.config.subtasks:
            try:
                result = self.run_subtask(subtask_config)
                self.results[subtask_config['name']] = result
            except Exception as e:
                logger.error(f"Failed to run subtask {subtask_config['name']}: {e}", exc_info=True)
                self.results[subtask_config['name']] = {'error': str(e)}
            
        if self.config.output_config.get('aggregate_subtasks', False):
            self.results['aggregate'] = self.aggregate_results()
            
        self.save_results()
        return self.results
        
    def aggregate_results(self) -> Dict:
        """Aggregate metrics across subtasks"""
        aggregate = {}
        all_metrics = {}
        
        for task_name, result in self.results.items():
            if 'metrics' in result:
                for metric_name, score in result['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(score)
                
        for metric_name, scores in all_metrics.items():
            aggregate[f"avg_{metric_name}"] = sum(scores) / len(scores)
            
        return aggregate
        
    def save_results(self):
        """Save results to file"""
        output_dir = Path(self.config.output_config.get('output_dir', 'results/raw'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_slug = self.model.model_name.replace('/', '_')
        output_path = output_dir / f"{self.config.name}_{model_name_slug}_{timestamp}.json"
        
        # Convert all numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert_numpy)
            
        logger.info(f"Results saved to {output_path}")
