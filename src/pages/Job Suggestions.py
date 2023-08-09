import os
import streamlit as st
import pandas as pd
from langchain.llms import HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
st.title(":white[Job Suggestions]")

job_desc = st.text_input("Enter Job Description")
repo_id = "tiiuae/falcon-7b"
submit = st.button("Get Job titles")
submitted = st.button("Get alternate Job Description")

# The code block you provided is checking if the "submit" button has been clicked. If it has, it sets
# up a prompt template for generating job titles based on the provided job description. It then uses a
# language model to generate the job titles and displays them in a dataframe.
if submit:

    template = """Prompt {question}
    Sure here are some of the titles that match the description"""
    question = "Generate some proper job titles for the given description: {0}".format(job_desc)

# The code block you provided is checking if the "submitted" button has been clicked. If it has, it
# sets up a prompt template for generating an alternate job description based on the provided job
# description. It then uses a language model to generate the alternate job description and displays it
# in a code block. The generated alternate job description is appended to the existing template in
# each iteration of the while loop, up to a maximum of 5 iterations.
elif submitted:
    # 
    template = """Prompt: {question} Sure here is an alternate version"""
    question = "Provide an alternate job description for the provided description {0}".format(job_desc)

# The code block you provided is checking if the "submit" button has been clicked. If it has, it sets
# up a prompt template for generating job titles based on the provided job description. It then uses a
# language model to generate the job titles and displays them in a dataframe.
if submit:
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "top-k": 50, "top-p":.85, "min_new_tokens": 1024}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    text = llm_chain.run(question)
    if submit:
        text_list = [x for x in text.split("\n") if len(x) > 2]
        templatedf = pd.DataFrame(text_list)
        templatedf.columns = ['Job titles']
        st.markdown("**Some suggested job titles**")
        st.dataframe(templatedf)

# The code block you provided is checking if the "submitted" button has been clicked. If it has, it
# sets up a prompt template for generating an alternate job description based on the provided job
# description.
if submitted:
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
