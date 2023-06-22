from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ChatVectorDBChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

from dotenv import load_dotenv, find_dotenv
from metapub import PubMedFetcher
import pandas as pd

# Load API keys from local environment
def load_api_keys():
    load_dotenv(find_dotenv())


def get_pubmed_articles(query,max_nr_results):
    """Get PubMed results"""
    fetch = PubMedFetcher()
    pm_ids = fetch.pmids_for_query(query, retmax=max_nr_results)

    return pm_ids

def get_details(pm_ids):
    """Get details from PubMed article ID"""
    fetch = PubMedFetcher()
    articles = pd.DataFrame(columns=['pmid','title','year','abstract','citation','url'])

    for pmid in range(len(pm_ids)):
        articles.loc[pmid] = [pm_ids[pmid],
                              fetch.article_by_pmid(pm_ids[pmid]).title,
                              fetch.article_by_pmid(pm_ids[pmid]).year,
                              fetch.article_by_pmid(pm_ids[pmid]).abstract,
                              fetch.article_by_pmid(pm_ids[pmid]).citation,
                              "https://pubmed.ncbi.nlm.nih.gov/"+pm_ids[pmid]+"/"]
    return articles


def text_split(text):
    """Split text into chunks of fixed size"""
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2048,
    chunk_overlap  = 20,
    length_function = len,
    )

    #text_chunks = text_splitter.create_documents([text])
    text_chunks = text_splitter.split_documents(text)

    return text_chunks


def get_query_from_question(question, model_name, openai_api_key):
    """Get a PubMed search query from a question"""
    template = """Given a question, your task is to come up with a relevant search term that would retrieve relevant articles from a scientific article database. The search term should not be so specific as to be unlikely to retrieve any articles, but should also not be so general as to retrieve too many articles. The search term should be a single word or phrase, and should not contain any punctuation. Convert any initialisms to their full form.
    Question: What are some treatments for diabetic macular edema?
    Search Query: diabetic macular edema
    Question: What is the workup for a patient with a suspected pulmonary embolism?
    Search Query: pulmonary embolism treatment
    Question: What is the recommended treatment for a grade 2 PCL tear?
    Search Query: posterior cruciate ligament tear
    Question: What are the possible complications associated with type 1 diabetes and how does it impact the eyes?
    Search Query: type 1 diabetes eyes
    Question: When is an MRI recommended for a concussion?
    Search Query: concussion magnetic resonance imaging
    Question: {question}
    Search Query: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key))
    query = llm_chain.run(question)

    return query

def get_abstracts(articles):
    docs = []
    for a in range(len(articles)):
        docs.append(Document(page_content=articles['abstract'][a], 
                             metadata={"citation": articles['citation'][a],
                                       "url": articles['citation'][a] +" [link]("+ articles['url'][a] + ")",
                                       "pmid": articles['pmid'][a]}))
    
    return docs
