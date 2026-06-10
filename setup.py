from setuptools import setup, find_packages

setup(
    name="proactivegym",
    version="1.0.0",
    description="A Gymnasium environment for training proactive agents that anticipate user needs",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "openai",
        "numpy",
    ],
    include_package_data=True,
)
