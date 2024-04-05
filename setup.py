from setuptools import setup

setup(
    name="openparse",
    version="0.4.0",
    entry_points={
        "console_scripts": [
            "openparse-download= openparse.cli:download_unitable_weights",
        ],
    },
    install_requires=[
        "PyMuPDF >= 1.23.2",
        "pillow >= 8.3",
        "pydantic >= 2.0",
        "pypdf >= 4.0.0",
        "pdfminer.six >= 20200401",
        "tiktoken >= 0.3",
    ],
    extras_require={
        "ml": [
            "torch",
            "torchvision",
            "transformers",
            "tokenizers",
        ],
    },
    author="Sergey Filimonov",
    author_email="hello@sergey.fyi",
    description="Streamlines the process of preparing documents for LLM's.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Filimoa/open-parse/",
)
