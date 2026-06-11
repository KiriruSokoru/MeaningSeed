cat > setup.py << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meaning_seed",
    version="0.1.0",
    author="Kirill Sokol",
    author_email="kirill.sokol@example.com",
    description="Topological orchestration of LLM tasks via Ricci curvature",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KiriruSokoru/MeaningSeed",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "networkx>=3.0",
        "GraphRicciCurvature>=0.5",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        'console_scripts': [
            'meaningseed=cli:main',
        ],
    },
)
EOF
