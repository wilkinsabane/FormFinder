"""Setup configuration for FormFinder package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "#" in line:
            # Remove inline comments
            line = line.split("#")[0].strip()
        if line and not line.startswith("#"):
            requirements.append(line)

# Separate core requirements from development requirements
core_requirements = []
dev_requirements = []
dev_section = False

for req in requirements:
    if "Development dependencies" in req or "Security and code quality" in req or "Testing utilities" in req or "Documentation" in req:
        dev_section = True
        continue
    
    if dev_section:
        dev_requirements.append(req)
    else:
        core_requirements.append(req)

setup(
    name="formfinder",
    version="2.0.0",
    author="FormFinder Team",
    author_email="contact@formfinder.com",
    description="A comprehensive football prediction system with database storage and workflow orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FormFinder",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Sports Enthusiasts",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Games/Entertainment :: Sports",
    ],
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
        "postgresql": ["psycopg2-binary>=2.9.9"],
        "sms": ["twilio>=9.0.4"],
        "docs": [
            "sphinx>=7.3.7",
            "sphinx-rtd-theme>=2.0.0",
            "mkdocs>=1.6.0",
            "mkdocs-material>=9.5.18",
        ],
        "monitoring": [
            "psutil>=5.9.8",
            "memory-profiler>=0.61.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "formfinder=formfinder.cli:main",
            "formfinder-workflow=formfinder.workflows:cli",
            "formfinder-db=formfinder.database:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "formfinder": [
            "config/*.yaml",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "football",
        "soccer",
        "prediction",
        "machine learning",
        "sports analytics",
        "data pipeline",
        "workflow orchestration",
        "database",
        "api",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/FormFinder/issues",
        "Source": "https://github.com/yourusername/FormFinder",
        "Documentation": "https://formfinder.readthedocs.io/",
        "Changelog": "https://github.com/yourusername/FormFinder/blob/main/CHANGELOG.md",
    },
)