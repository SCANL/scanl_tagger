from setuptools import setup, find_packages

with open('version.py') as f:
    exec(f.read())

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="scanl_tagger",
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'scanl_tagger=scanl_tagger.main:main',
        ],
    },
    python_requires='>=3.12',
    author="Christian Newman, Anthony Peruma, Brandon Scholten, Syreen Banabilah",
    description="A machine learning based tagger for source code analysis",
)