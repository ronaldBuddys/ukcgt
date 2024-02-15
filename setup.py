from setuptools import setup, find_packages


setup(
    name='capitalgains',
    version='0.0.1',
    author='ronaldBuddys',
    author_email='ronaldbuddys@gmail.com',
    description='For UK capital gains calculations (CGT).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ronalbuddys/capitalgains',
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)