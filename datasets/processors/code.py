import re
from typing import List

class CodeProcessor:
    """Utilities for preprocessing source code."""

    def __init__(self, language: str = 'python'):
        """
        Initializes the code processor.

        Args:
            language: The programming language to process.
        """
        self.language = language

    def remove_comments(self, code: str) -> str:
        """Removes comments from a block of code."""
        if self.language == 'python':
            # Remove single-line comments
            code = re.sub(r'#.*?\n', '\n', code)
            # Remove multi-line comments (docstrings)
            code = re.sub(r'(""".*?""")|('''.*?''')', '', code)
        # Add rules for other languages here
        return code

    def normalize_whitespace(self, code: str) -> str:
        """Normalizes whitespace in the code (e.g., multiple newlines)."""
        # Replace multiple newlines with a single one
        code = re.sub(r'\n+', '\n', code)
        # Replace multiple spaces with a single one, except for indentation
        lines = code.split('\n')
        processed_lines = []
        for line in lines:
            leading_whitespace = len(line) - len(line.lstrip(' '))
            processed_line = ' ' * leading_whitespace + ' '.join(line.lstrip().split())
            processed_lines.append(processed_line)
        return '\n'.join(processed_lines)

    def process(self, code: str, remove_comments: bool = True, normalize_whitespace: bool = True) -> str:
        """Applies a full preprocessing pipeline to a code string."""
        if remove_comments:
            code = self.remove_comments(code)
        if normalize_whitespace:
            code = self.normalize_whitespace(code)
        return code.strip()

    def batch_process(self, code_list: List[str], **kwargs) -> List[str]:
        """Applies the processing pipeline to a list of code strings."""
        return [self.process(code, **kwargs) for code in code_list]
