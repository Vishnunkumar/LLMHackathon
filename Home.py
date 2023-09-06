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
    
    job_result = np.array(embeddings.embed_query(job_desc)).reshape(1, -1)
    job_ttl = np.array(embeddings.embed_query(job_title)).reshape(1, -1)
    return dist.pairwise(job_result, job_ttl)[0][0]

job_desc = st.text_input("Enter Job description")
job_title = st.text_input("Enter Job Title")

if job_desc and job_title:
    st.markdown("_The distance is:_")
    st.success(str(score_job_tile(job_desc, job_title)))
