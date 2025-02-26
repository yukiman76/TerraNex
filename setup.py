from setuptools import setup, find_packages

setup(
    name="humanityhelper",
    version="0.1.0",
    author="Jeffrey Rivero",
    author_email="jeff@check-ai.com",
    description="MoE language model with domain-specific expertise",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-github/HumanityHelper",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "mlflow>=1.20.0",
        "datasets>=1.12.0",
        "pyyaml>=5.1",
    ],
    entry_points={
        "console_scripts": [
            "humanityhelper-train=src.train:main",
        ],
    },
)
