import os
import json
import smtplib
import streamlit as st
import numpy as np
import pandas as pd
from pdfreader import SimplePDFViewer
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics import DistanceMetric

# The code `f = open('config.json', encoding="utf-8")` opens the file `config.json` in read mode with
# the specified encoding of "utf-8".
f = open('config.json', encoding="utf-8")
data = json.load(f)

s_email = data['sender_email']
s_pass = data['sender_password']

embeddings = HuggingFaceEmbeddings()
dist = DistanceMetric.get_metric('euclidean')

cvs_list = []

def score_job_tile(job_desc, cv_content):
    """
    The function `score_job_tile` calculates the similarity score between a job description and a CV
    content using embeddings and pairwise distance.
    
    :param job_desc: The job description, which is a string that describes the requirements and
    responsibilities of a job position. It provides information about the skills, qualifications, and
    experience needed for the job
    :param cv_content: The `cv_content` parameter represents the content of a candidate's CV (Curriculum
    Vitae), which typically includes information about their education, work experience, skills, and
    other relevant details
    :return: the score or similarity between the job description and the content of a CV.
    """

    job_result = np.array(embeddings.embed_query(job_desc)).reshape(1, -1)
    job_ttl = np.array(embeddings.embed_query(cv_content)).reshape(1, -1)
    return dist.pairwise(job_result, job_ttl)[0][0]

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

st.title(":green[CV Ranker]")
st.markdown("_Rank CVs based on the Job Description_")

st.markdown("### Provide Job Description")
job_desc = st.text_area("Job Description")

st.markdown("### Upload CVs")
cvs = st.file_uploader("Chose multiple CVs or single CV", accept_multiple_files=True)
cvs_sub = st.button("Load CVs")

# The code block `if cvs_sub:` checks if the "Load CVs" button has been clicked. If it has, then it
# iterates over each uploaded CV file (`fil_`) and performs the following steps:
if cvs_sub:
    for fil_ in cvs:
        cv_json = {}
        viewer = SimplePDFViewer(fil_)
        viewer.render()
        cv_json['title'] = viewer.metadata['Title']
        cv_json['content'] = ("").join(viewer.canvas.strings)
        cv_json['score'] = score_job_tile(job_desc, cv_json['content'])
        cv_json['email'] = "vishnunkumar25@gmail.com"
        cvs_list.append(cv_json)
    
    cv_df = pd.DataFrame(cvs_list)
    cv_df.sort_values(by=['score'], inplace=True)
    st.dataframe(cv_df.iloc[:5,:])

 # This code block is responsible for sending emails to the top 5 candidates who have been shortlisted
 # based on their CV scores.
    emails = st.button("Send Emails")
    if emails:
        for i in range(cv_df.iloc[:5,:]):
            st.info(cv_df.shape[0])
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls()
            s.login(s_email, s_pass)
            MESSAGE = "You are shortlisted"
            s.set_debuglevel(True)
            s.sendmail(s_email, cv_df['email'].iloc[i], MESSAGE)
            s.quit()
        
        st.success("Emails Sent successfully")
