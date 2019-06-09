# Detecting Hateful Speech in Social Media Comments

In this project, we apply machine learning to unstructured data to detect hate speech in comments from the [Civil Comments dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), with labeling informed by the Online Hate Index Research Project at D-Lab, University of California, Berkeley.

### Goal
Our goal is to classify comments as hateful or not hateful. Historically, attempts to do similar classifications misidentify comments that mention identify groups that could be attacked with hate speech as hateful. We hope to develop more nuanced models that correctly categorize both hateful speech and non-hateful identity references.

### Team Members
- [Andrea Koch](https://github.com/kochandrea)
- [Natasha Mathur](https://github.com/natashamathur)
- [Loren Hinkson](https://github.com/lorenh516)

### Technologies
Python:
  - [Pytorch](https://pytorch.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [pandas](https://pandas.pydata.org/)
  - [NLTK](https://www.nltk.org/)

Amazon Web Services:
  - [S3 buckets](https://aws.amazon.com/s3/)
  - [SageMaker](https://aws.amazon.com/sagemaker/)

Google Cloud Services:
  - [Google CoLaboratory](https://colab.research.google.com/)
  - [Google Cloud Storage](https://cloud.google.com/storage/)

### Files & Notebooks
#### Final Models
- [NB_final.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/NB_final.ipynb) Naive Bayes Model (698 lines)
- [SVM_final.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/SVM_final.ipynb) Support Vector Machines Model (1818 lines)
- [neural_network.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/neural_network.ipynb) Two Layer Neural Network (536 lines)
- [final_lstm.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/final_lstm.ipynb) Three Layer Bidirectional Long Short-Term Memory Recurrent Neural Network (7514 lines)
#### Feature Generation
- [feature_generation_functions.py](https://github.com/natashamathur/no_hate_all_love/blob/master/feature_generation_functions.py):  Contains modules and functions used to generate text and numerical features for model. (273 lines)
- [feature_generation.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/feature_generation.ipynb):  Python 3 notebook used to run functions from feature_generation_functions.py and pickle_functions.py.  Generates features, pickles data frames, and sends to s3 bucket. (160 lines)
#### Helper Functions
- [model_functions.py](https://github.com/natashamathur/no_hate_all_love/blob/master/model_functions.py): Contains modules and functions to generate and test Naive Bayes and SVM models; run metrics on models. (226 lines)
- [pickle_functions.py](https://github.com/natashamathur/no_hate_all_love/blob/master/pickle_functions.py):  Contains modules and functions used to read/write data from/to pickle files hosted in AWS s3 bucket. (60 lines)
- [exploration/exploration_functions.py](https://github.com/natashamathur/no_hate_all_love/blob/master/exploration/exploration_functions.py): Contains modules and functions used to explore dataset. (103 lines)
#### Intermediate Models
- [Stepping_Stones](https://github.com/natashamathur/no_hate_all_love/tree/master/stepping_stones): Iterations of each model that was built prior to the final model design and assessment
  - [Initial_Models_Exploration.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/Initial_Models_Exploration.ipynb) (1697 lines)
  - [NB_iter1.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/NB_iter2.ipynb) (726 lines)
  - [NB_iter2.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/NB_iter3.ipynb) (626 lines)
  - [NB_iter3.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/NB_iter4.ipynb) (865 lines)
  - [SVM_iter1.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/SVM_iter1.ipynb) (657 lines)
  - [SVM_iter2.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/SVM_iter2.ipynb) (691 lines)
  - [SVM iter3.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/SVM_iter3.ipynb) (644 lines)
  - [initial_lstm.ipynb](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/initial_lstm.ipynb) (1920 lines)
  - [exec_lstm](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/exec_lstm.py) (587 lines) and [rcc_run_model.sh](https://github.com/natashamathur/no_hate_all_love/blob/master/stepping_stones/rcc_run_model.sh) (27 lines)


If there are any issues opening a notebook, please enter the link into the renderer at the following site: https://nbviewer.jupyter.org/

#### [Final Report](https://github.com/natashamathur/no_hate_all_love/blob/master/A%20Machine%20Learning%20Approach%20to%20Intervening%20on%20Toxic%20Comments%20in%20Online%20Forums.pdf)
