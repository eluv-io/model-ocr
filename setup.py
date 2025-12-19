from setuptools import setup

setup(
    name='my_package',
    version='0.1',
    packages=['ocr'],
    install_requires=[
        'easydict==1.9',
        'opencv_python==4.8.0.76',
        'scikit-image==0.17.2',
        'torch==1.9.0',
        'torchvision==0.10.0',
        'numpy==1.21.6',
        'setproctitle',
        'common_ml @ git+https://github.com/eluv-io/common-ml.git#egg=common_ml',
    ]
)
