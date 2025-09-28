from setuptools import setup, find_packages

setup(
    name="universal-model-benchmarks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface-hub",
        "datasets",
        "transformers",
        "scikit-learn",
        "pandas",
        "numpy",
        "torch",
        "torchvision",
        "Pillow",
        "pyyaml",
        "rouge_score",
        "nltk",
    ],
)
