from distutils.core import setup


setup(
  name='micromobilenet',
  packages=['micromobilenet'],
  version='1.0.0',
  license='MIT',
  description='Variations of MobileNetV1 for emdedded CPUs',
  author='Simone Salerno',
  author_email='support@eloquentarduino.com',
  url='https://github.com/eloquentarduino/micromobilenet',
  keywords=[
    'ML',
    'Edge AI'
  ],
  install_requires=[
    'numpy',
    'keras',
    'tensorflow',
    'Jinja2',
    'cached_property'
  ]
)