# Detecting Hateful Speech in Social Media Comments

In this project, we apply machine learning to unstructured data to detect hate speech in comments from the [Civil Comments dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), with labeling informed by the Online Hate Index Research Project at D-Lab, University of California, Berkeley.

### Goal
Our goal is to classify comments as hateful or not hateful. Historically, attempts to do similar classifications mis-identify comments that mention identify groups that could be attacked with hate speech as hateful. We hope to develop more nuanced models that correctly categorize both hateful speech and non-hateful identity references.

### Team Members
- [Andrea Koch](https://github.com/kochandrea)
- [Natasha Mathur](https://github.com/lorenh516)
- [Loren Hinkson](https://github.com/natashamathur)

### Technologies
Python:
  - [Pytorch](https://pytorch.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [pandas](https://pandas.pydata.org/)
  - [NLTK](https://www.nltk.org/)
  
Amazon Web Services:
  - [S3 buckets](https://aws.amazon.com/s3/)
  - [SageMaker](https://aws.amazon.com/sagemaker/)

### Files & Notebooks
- feature_generation_functions.py:  Contains modules and functions used to generate text and numerical features for model. (229 lines)
- pickle_functions.py:  Contains modules and functions used to read/write data from/to pickle files hosted in AWS s3 bucket. (50 lines)
- feature_generation.ipynb:  Python 3 notebook used to run functions from feature_generation_functions.py and pickle_functions.py.  Generates features, pickles data frames, and sends to s3 bucket. (160 lines)

If there are any issues opening a notebook, please enter the link at the following site: https://nbviewer.jupyter.org/
