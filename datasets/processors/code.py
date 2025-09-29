import re
from typing import List

class CodeProcessor:
    """Processes code data."""

    def __init__(self, language: str, remove_comments: bool = False):
        self.language = language
        self.remove_comments = remove_comments

    def process(self, code: str) -> str:
        """
        Processes a single code snippet.

        Args:
            code: The code string to process.

        Returns:
            The processed code string.
        """
        if self.remove_comments:
            code = self._strip_comments(code)
        
        return code

    def _strip_comments(self, code: str) -> str:
        """Strips comments from a code string based on the language."""
        if self.language == 'python':
            # Remove single-line comments
            code = re.sub(r'#.*\n', '\n', code)
            # Remove multi-line comments (docstrings)
            code = re.sub(r'"""[\s\S]*?"""', '', code)
            code = re.sub(r"''[\s\S]*?'''", '', code)
        # Add more languages as needed
        
        return code

    def batch_process(self, code_snippets: List[str]) -> List[str]:
        """Processes a batch of code snippets."""
        return [self.process(snippet) for snippet in code_snippets]