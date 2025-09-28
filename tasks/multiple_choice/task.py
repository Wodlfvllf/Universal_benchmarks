from tasks.base import BaseTask, TaskInput, TaskOutput
from typing import List, Dict, Any

class MultipleChoiceQATask(BaseTask):
    """Implementation for multiple choice question answering"""
    
    def setup(self):
        self.choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """Prepare multiple choice inputs"""
        inputs = []
        
        for example in dataset:
            # Standard format: question, choices, answer
            question = example['question']
            choices = example['choices']
            
            # Add context if available
            context = example.get('context', '')
            
            # Get correct answer index
            answer = example.get('answer', None)
            if isinstance(answer, str) and answer in self.choice_labels:
                answer = self.choice_labels.index(answer)
                
            inputs.append(TaskInput(
                data={
                    'question': question,
                    'choices': choices,
                    'context': context
                },
                labels=answer
            ))
            
        return inputs
    
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format multiple choice prompt"""
        data = input_data.data
        
        prompt_parts = []
        
        # Add context if present
        if data['context']:
            prompt_parts.append(f"Context: {data['context']}\n")
            
        # Add question
        prompt_parts.append(f"Question: {data['question']}\n")
        
        # Add choices
        prompt_parts.append("Choices:\n")
        for i, choice in enumerate(data['choices']):
            prompt_parts.append(f"{self.choice_labels[i]}) {choice}\n")
            
        # Add instruction
        prompt_parts.append("\nAnswer (A/B/C/D/...):")
        
        return "".join(prompt_parts)
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Extract choice from output"""
        # Clean output
        output = raw_output.strip().upper()
        
        # Look for choice label
        for i, label in enumerate(self.choice_labels):
            if label in output[:10]:  # Check beginning of output
                return TaskOutput(predictions=i)
                
        # Default to first choice if unclear
        return TaskOutput(predictions=0)

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """Generate predictions using model"""
        pass

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute task-specific metrics"""
        pass