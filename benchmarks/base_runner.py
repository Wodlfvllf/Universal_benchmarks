from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from datasets import load_dataset
from tasks.registry import TaskRegistry
from metrics.registry import MetricRegistry
from models.registry import ModelRegistry
from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    name: str
    version: str
    dataset_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    output_config: Dict[str, Any]
    subtasks: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            name=config['benchmark']['name'],
            version=config['benchmark']['version'],
            subtasks=config.get('subtasks'),
            dataset_config=config['dataset'],
            evaluation_config=config['evaluation'],
            output_config=config['output']
        )

class BenchmarkRunner:
    """Base class for benchmark execution"""
    
    def __init__(self, config_path: str, model_name: str, model_config: Optional[Dict] = None):
        self.config = BenchmarkConfig.from_yaml(config_path)
        self.model = ModelRegistry.get_model(model_name, model_config)
        self.results = {}
        
    def load_data(self, subtask_config: Dict) -> Dict:
        """Load dataset for a subtask"""
        return DatasetRegistry.get_dataset(
            self.config.dataset_config['name'],
            name=subtask_config['dataset_config'],
            cache_dir=self.config.dataset_config.get('cache_dir')
        )
        
    def run_subtask(self, subtask_config: Dict, split: str = 'validation') -> Dict:
        """Execute a single subtask"""
        logger.info(f"Running subtask: {subtask_config['name']}")
        
        # Load task implementation
        task = TaskRegistry.get_task(subtask_config['task_type'])
        
        # Load dataset
        dataset = self.load_data(subtask_config)
        
        # Prepare inputs
        inputs = task.prepare_inputs(
            dataset[split], 
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
        metrics = {}
        references = [inp.labels for inp in inputs]
        for metric_name in subtask_config['metrics']:
            metric_fn = MetricRegistry.get_metric(metric_name)
            result = metric_fn.compute(predictions=[p.predictions for p in predictions], references=references)
            metrics[metric_name] = result.score
            
        return {
            'task': subtask_config['name'],
            'metrics': metrics,
            'predictions': [p.predictions for p in predictions] if self.config.output_config.get('save_raw_predictions') else None
        }
        
    def run(self) -> Dict:
        """Execute all subtasks in the benchmark"""
        logger.info(f"Starting benchmark: {self.config.name} v{self.config.version}")
        
        if self.config.subtasks:
            for subtask_config in self.config.subtasks:
                result = self.run_subtask(subtask_config)
                self.results[subtask_config['name']] = result
        else:
            # Treat the benchmark as a single task
            result = self.run_subtask(self.config.dataset_config)
            self.results[self.config.name] = result
            
        if self.config.output_config.get('aggregate_subtasks') and self.config.subtasks:
            self.results['aggregate'] = self.aggregate_results()
            
        self.save_results()
        return self.results
        
    def aggregate_results(self) -> Dict:
        """Aggregate metrics across subtasks"""
        aggregate = {}
        all_metrics = {}
        
        for task_name, result in self.results.items():
            if task_name == 'aggregate': continue
            for metric_name, score in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(score)
                
        for metric_name, scores in all_metrics.items():
            aggregate[f"avg_{metric_name}"] = sum(scores) / len(scores)
            
        return aggregate
        
    def save_results(self):
        """Save results to file"""
        import json
        from datetime import datetime
        
        output_dir = self.config.output_config.get('output_dir', 'results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model.config.model_name.replace('/', '_')
        output_path = Path(f"{output_dir}/{self.config.name}_{model_name}_{timestamp}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")