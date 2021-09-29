from setuptools import setup, find_packages
setup(
    name='pyquantification',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        # List packages and versions you depend on.
        'scikit-learn==0.24.2',
        'scipy==1.5.2',
        'plotly==4.14.3',
        'numpy==1.20.0',
        'pandas==1.2.4',
        'requests==2.24.0',
    ],
    extras_require={
        # Best practice to list non-essential dev dependencies here.
        'dev': [
            'flake8==3.9.2',
            'mypy==0.812',
            'pytest==5.2.2',
            'pytest-cov==2.8.1',
            # Typing stubs.
            'pandas-stubs==1.1.0.7',
        ]
    }
)
