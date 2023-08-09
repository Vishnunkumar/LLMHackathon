import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics import DistanceMetric

st.title(":blue[HR Buddy]")
st.subheader("Job Scorer")
embeddings = HuggingFaceEmbeddings()
dist = DistanceMetric.get_metric('euclidean')

def score_job_tile(job_desc, job_title):
    """
    The function `score_job_tile` calculates the similarity score between a job description and a job
    title using embeddings and pairwise distance.
    
    :param job_desc: The job description, which is a string that describes the responsibilities,
    requirements, and qualifications for a specific job position
    :param job_title: The job title is the title or name of the job position. It is typically a short
    and concise description of the role, such as "Software Engineer" or "Marketing Manager"
    :return: The function `score_job_tile` returns the pairwise distance between the embeddings of the
    job description (`job_desc`) and the job title (`job_title`).
    """
    
    job_result = np.array(embeddings.embed_query(job_desc)).reshape(1, -1)
    job_ttl = np.array(embeddings.embed_query(job_title)).reshape(1, -1)
    return dist.pairwise(job_result, job_ttl)[0][0]

job_desc = st.text_input("Enter Job description")
job_title = st.text_input("Enter Job Title")

# The code block `if job_desc and job_title:` checks if both `job_desc` and `job_title` variables have
# values. If both variables have values (i.e., they are not empty), it executes the following code:
if job_desc and job_title:
    st.markdown("_The distance is:_")
    st.success(str(score_job_tile(job_desc, job_title)))
