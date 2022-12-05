from setuptools import setup
setup(
   name='zeroshotclassifierexplainer',
   version='1.0',
   description='Zero-shot classification and LIME explanations.',
   author='Thorsten Zylowski',
   author_email='thorsten.zylowski@h-ka.de',
   packages=['zeroshotclassifierexplainer'],  #same as name
   install_requires=['transformers', 'lime'], #external packages as dependencies
)