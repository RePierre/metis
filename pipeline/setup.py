from setuptools import setup, find_packages

version = '0.1'

install_requires = [
    "lxml",
    "requests",
    "python-dateutil",
    "feedparser",
    "dict2xml"
]

tests_requires = [
    "pytest",
    "mock",
]

setup(
    name='data_pipeline',
    version=version,
    description="Packages for data pipeline.",
    long_description="",
    classifiers=[],  # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords='',
    author='',
    author_email='',
    url='',
    license='',
    packages=find_packages(exclude=['']),
    include_package_data=True,
    zip_safe=True,
    install_requires=install_requires,
    tests_require=tests_requires,
    test_suite="py.test",
)
