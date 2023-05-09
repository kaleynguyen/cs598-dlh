from setuptools import find_packages, setup

setup(
    name="mimic_iii",
    version="0.0.1",
    description="Events data modeling in MIMICIII",
    author="Kaley Nguyen",
    author_email="kaleynn7@gmail.com",
    packages=find_packages(exclude=["mimic_iii_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "polars",
        "numpy",
        "pytorch"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)
