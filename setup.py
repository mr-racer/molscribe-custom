from setuptools import setup

extras = {
    # training / evaluation code (dataset.py, augment.py, loss.py, train.py, evaluate.py)
    'train': ['pandas', 'albumentations==1.1.0', 'scipy', 'tensorboardX', 'transformers>=4.5.1',
              'matplotlib>=3.5.3'],
    'draw': ['matplotlib>=3.5.3'],               # MolScribe.draw_prediction
    'serve': ['fastapi', 'pydantic'],            # molscribe.remote.make_router
}
extras['all'] = sorted({req for reqs in extras.values() for req in reqs})

setup(name='MolScribe',
      version='1.1.1',
      description='MolScribe (custom fork: faster inference, larger label dictionary)',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://gitlab.odanchem.org/odanchem/molscribe-custom',
      packages=['molscribe', 'molscribe.indigo', 'molscribe.inference', 'molscribe.transformer'],
      package_dir={'molscribe': 'molscribe'},
      package_data={'molscribe': ['vocab/*']},
      python_requires='>=3.7',
      # inference only; everything else is an extra
      install_requires=[
        "numpy>=1.19.5",
        "torch>=1.11.0",
        "opencv-python>=4.5.5.64",
        "SmilesPE==0.0.3",
        "rdkit>=2022.3.3",
        "timm==0.4.12"
      ],
      extras_require=extras)
