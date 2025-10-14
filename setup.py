#!/usr/bin/env python3
"""
Aura Render - AI-Powered Video Generation Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "项目分析文档.md").read_text(encoding='utf-8')

setup(
    name="aura-render",
    version="0.1.0",
    description="AI-Powered Video Generation and Rendering Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aura Render Team",
    author_email="team@aurarender.com",
    url="https://github.com/yourusername/aura-render",
    
    packages=find_packages(),
    python_requires=">=3.8",
    
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0", 
        "pydantic>=2.5.0",
        "httpx>=0.25.0",
        "aiohttp>=3.9.0",
        "requests>=2.31.0",
        "dashscope>=1.17.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.22.0",
        "matplotlib>=3.8.0",
        "json5>=0.9.0",
        "python-dotenv>=1.0.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
        "video": [
            "moviepy>=1.0.3",
            "ffmpeg-python>=0.2.0",
        ],
        "queue": [
            "redis>=5.0.0", 
            "celery>=5.3.0",
        ],
        "db": [
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "aura-render=app:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    keywords="ai video generation rendering pipeline automation",
    include_package_data=True,
    zip_safe=False,
)