# Detecting Hateful Speech in Social Media Comments

In this project, we apply machine learning to unstructured data to detect hate speech in comments from the [Civil Comments dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), with labeling informed by the Online Hate Index Research Project at D-Lab, University of California, Berkeley.

### Goal
Our goal is to classify comments as hateful or not hateful. Historically, attempts to do similar classifications mis-identify comments that mention identify groups that could be attacked with hate speech as hateful. We hope to develop more nuanced models that correctly categorize both hateful speech and non-hateful identity references.

### Team Members
- Andrea Koch
- Natasha Mathur
- Loren Hinkson

### Technologies
Python:
  - Pytorch
  - scikit-learn
  - pandas
  - NLTK
  
Amazon Web Services:
  - S3 buckets
  - SageMaker

### Files & Notebooks
- feature_generation_functions.py:  Contains modules and functions used to generate text and numerical features for model. (186 lines)
- pickle_functions.py:  Contains modules and functions used to read/write data from/to pickle files hosted in AWS s3 bucket. (50 lines)
- feature_generation.ipynb:  Python 3 notebook used to run functions from feature_generation_functions.py and pickle_functions.py.  Generates features, pickles data frames, and sends to s3 bucket. (160 lines)
