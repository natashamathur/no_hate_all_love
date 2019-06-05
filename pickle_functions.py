###############
#   MODULES   #
###############

import io
import boto3
# import gcsfs
import pickle
import _pickle as cPickle
import pandas as pd
import boto3.session

#################
#   FUNCTIONS   #
#################

def write_pickle_to_s3bucket(filename, df, bucket_name):
    '''
    Pickles a data frame and sends to AWS s3 bucket.

    Input(s):
        filename      - (string) the filename for the pickle
        df            - (data frame) the frame you want to pickle
        bucket_name   - (string) the name of the s3 bucket into which
                            pickle file will be dumped
    '''
    pickle_buffer = io.BytesIO()
    s3_resource = boto3.resource('s3')
    bucket = bucket_name
    key = filename+'.pkl'

    df.to_pickle(key)
    s3_resource.Object(bucket,key).put(Body=open(key, 'rb'))
    print("Pickled and sent to bucket!")




def read_pickle(bucket_name, filename):
    '''
    Reads a pickled object from s3 bucket and puts into pandas data frame.

    Input(s):
        filename      - (string) the pickle object filename
        bucket_name   - (string) s3 bucket name

    Output(s):
        df - (data frame) Pandas data frame of pickle file

    Code modified from: https://github.com/noopurrkalawatia/Dhvani/blob/master/Boto3_features.ipynb
    '''
    session = boto3.session.Session(region_name='us-east-1')
    s3client = session.client('s3')

    response = s3client.get_object(Bucket=bucket_name, Key=filename+'.pkl')

    body_string = response['Body'].read()
    df = cPickle.loads(body_string)
    return df


def pickle_to_gcs(df, filename, bucket_name="no-hate", directory=None):
    pickle_buffer = io.BytesIO()
    fs = gcsfs.GCSFileSystem(project=project_id)
    df.to_pickle(filename)

    if directory:
        directory = directory + "/"
    else:
        directory = ""

    with fs.open(f"{bucket_name}/{directory}{filename}", "wb") as handle:
        pickle.dump(df, handle)


def load_pickle_from_gcs(filename, bucket_name="no-hate",  directory=None):
    fs = gcsfs.GCSFileSystem(project=project_id)

    if not directory:
        directory = directory + "/"
    else:
        directory = ""

    with fs.open(f"{bucket_name}/{directory}{filename}", "rb") as handle:
        df = pickle.load(handle)
    return df
