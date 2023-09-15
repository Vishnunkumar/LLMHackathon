import warnings
warnings.filterwarnings('ignore')

import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint, HuggingFaceHub, HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

st.title(":blue[ChatDF]")
st.text("Ask anything from the PDF shared and Voila! the bot will answer")

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
embeddings = HuggingFaceEmbeddings()
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"top-k": 50, "top-p":.90, "min_new_tokens": 256})

uploaded_file = st.file_uploader("Choose a file", "pdf")
if uploaded_file is not None:
    loader = PyPDFLoader("input/samplepdf/Vishnu_Nandakumar_Dev.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    st.text(qa.run(query))
