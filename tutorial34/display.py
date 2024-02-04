import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.vectorstores import FAISS
import base64
import os

st.set_page_config(page_title="COOKING ASSISTANT")
st.header = "write your query about indian cushine"
qsn = st.text_input("enter your question")

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

vs = FAISS.load_local("cushine_index_2", embeddings=OpenAIEmbeddings())

prompt_template = """
As a renowned chef with expertise in Indian cuisine, 
please answer all questions relying solely on the context provided, which may include text, images, and tables.
{context}
Question: {question}
Please provide a detailed and accurate response only if you are certain about the information.
If unsure, kindly decline by stating, "Sorry, I don't have much information about it."
Answer:
"""

qa_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4", max_tokens=1024),
    prompt=PromptTemplate.from_template(prompt_template)
)


def answer(qsn):
    rd = vs.similarity_search(qsn)
    context = ""
    ri = []
    for d in rd:
        if d.metadata["type"] == "text":
            context += '[text]' + d.metadata["original_content"]
        elif d.metadata["type"] == "table":
            context += '[table]' + d.metadata["original_content"]
        elif d.metadata["type"] == "image":
            context += '[image]' + d.page_content
            ri.append(d.metadata["original_content"])
    result = qa_chain.run({"context": context, "question": qsn})
    return result, ri

if st.button("Submit", type="primary"):
    if qsn is not None:
        result, ri = answer(qsn)
        st.write(result)
        x = ri[0].encode("utf-8")
        st.image(base64.b64decode(x))
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.chains import LLMChain,RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import base64,os


st.set_page_config(page_title="COOKING ASSISTANT")
st.header="write your query about indian cushine"
qsn=st.text_input("enter your qwuestion")
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
vs=FAISS.load_local("cushine_index_2",embeddings=OpenAIEmbeddings())
prompt_template = """
As a renowned chef with expertise in Indian cuisine, 
please answer all questions relying solely on the context provided, which may include text, images, and tables.
{context}
Question: {question}
Please provide a detailed and accurate response only if you are certain about the information.
If unsure, kindly decline by stating, "Sorry, I don't have much information about it."
Answer:
"""
qa_chain=LLMChain(
    llm=ChatOpenAI(model="gpt-4",max_tokens=1024),
    prompt=PromptTemplate.from_template(prompt_template)
)


def answer(qsn):
    rd=vs.similarity_search(qsn)
    context=""
    ri=[]
    for d in rd:
        if d.metadata["type"]=="text":
            context +='[text]' + d.metadata["original_content"]
        elif d.metadata["type"]=="table":
            context +='[table]' + d.metadata["original_content"]
        elif d.metadata["type"]=="image":
            context +='[image]' + d.page_content
            ri.append(d.metadata["original_content"])
    result=qa_chain.run({"context":context,"question":qsn})
    return result,ri

if st.button("Submit",type="primary"):
    if qsn is not None:
        result,ri=answer(qsn)
        st.write(result)
        x=ri[0].encode("utf-8")
        st.image(base64.b64decode(x))
