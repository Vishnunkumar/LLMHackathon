import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics import DistanceMetric
from langchain.llms import HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

st.title(":blue[HR Buddy]")
st.text("A space for HR executives to analyze and optimize the selection process")
embeddings = HuggingFaceEmbeddings()
dist = DistanceMetric.get_metric('euclidean')
repo_id = "tiiuae/falcon-7b"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

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
submit_one = st.button("Get comparison of the title and description")

# The code block `if job_desc and job_title:` checks if both `job_desc` and `job_title` variables have
# values. If both variables have values (i.e., they are not empty), it executes the following code:
if submit_one:
    if score_job_tile(job_desc, job_title) > 1.2:
        st.error("Not Matching")
    else:
        st.success("Matching")

title_submit = st.button("Get a proper Job Title")
if title_submit:
    template = """Prompt: {question} Sure here is the best job title for the given job description"""
    question = "Generate a proper job title for the given job description: {0}".format(job_desc)

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "top-k": 50, "top-p":.85, "min_new_tokens": 1024}
        )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    text = llm_chain.run(question)
    text_list = [x for x in text.split("\n") if len(x) > 2]
    templatedf = pd.DataFrame(text_list)
    templatedf.columns = ['Job titles']
    st.markdown("**Some suggested job titles**")
    st.dataframe(templatedf)

job_submit = st.button("Generate alternative Job descriptions")
if job_submit:
    template = """Prompt: {question} Sure here is an alternate version"""
    question = "Provide an alternate job description for the provided description {0}".format(job_desc)

    k = 0
    while k < 5:
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm = HuggingFaceHub(
            repo_id=repo_id, model_kwargs={"temperature": 0.5, "top-k": 50, "top-p":.85, "min_new_tokens": 1024}
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        text = llm_chain.run(question)
        template = template + "\n" + text  
        k = k + 1
    st.markdown("_Alernate Job description:_")
    st.code(template, language='textile')

