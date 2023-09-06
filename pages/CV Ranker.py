import os
import streamlit as st
import numpy as np
import pandas as pd
import pdfreader
from pdfreader import PDFDocument, SimplePDFViewer
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics import DistanceMetric

embeddings = HuggingFaceEmbeddings()
dist = DistanceMetric.get_metric('euclidean')

cvs_list = []

def score_job_tile(job_desc, cv_content):
    job_result = np.array(embeddings.embed_query(job_desc)).reshape(1, -1)
    job_ttl = np.array(embeddings.embed_query(cv_content)).reshape(1, -1)
    return dist.pairwise(job_result, job_ttl)[0][0]

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

st.title(":green[CV Ranker]")
st.markdown("_Rank CVs based on the Job Description_")

st.markdown("### Upload CVs")
cvs = st.file_uploader("Chose multiple CVs or single CV", accept_multiple_files=True)
cvs_sub = st.button("Load CVs")

st.markdown("### Provide Job Description")
job_desc = st.text_area("Job Description")
job_sub = st.button("Submit Job Description")

if cvs_sub:
    for fil_ in cvs:
        viewer = SimplePDFViewer(fil_)
        viewer.render()
        cvs_list.append(("").join(viewer.canvas.strings))

cvs_list = [score_job_tile(job_desc, x) for x in cvs_list]

get_ranks = st.button("Get first five")

if get_ranks:
    st.write(cvs_list)
