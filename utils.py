from langchain import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import requests
from metapub import PubMedFetcher
import pandas as pd


keyword = 'CKD'

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


a = get_pubmed_articles(keyword,3)
print(a)
b = get_details(a)
print(b)