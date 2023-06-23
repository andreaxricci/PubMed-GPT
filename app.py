import os 

import streamlit as st
from streamlit_chat import message

from utils import get_query_from_question, load_api_keys, get_pubmed_articles, get_details, text_split, get_abstracts, get_condensed_question, get_chat_answer

max_nr_results = 3
model = 'gpt-3.5-turbo'

st.set_page_config(
    page_title="PubMed Chat",
    page_icon=":heart:"
)

st.header("PubMed Chat")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text 

load_api_keys()
openai_api_key = os.getenv('OPENAI_API_KEY')

user_input = get_text()

if user_input:

    chat_history_tuples = [(st.session_state['past'][i], st.session_state['generated'][i]) for i in range(len(st.session_state['generated']))]
    condensed_question = get_condensed_question(user_input, chat_history_tuples, model, openai_api_key)
    query = get_query_from_question(condensed_question, model, openai_api_key)
    pm_ids = get_pubmed_articles(query,max_nr_results)
    if len(pm_ids) > 0:
        details = get_details(pm_ids)
        docs = get_abstracts(details)
        text_chunks = text_split(docs)
        result, cit = get_chat_answer(user_input, text_chunks, model, openai_api_key, st.session_state)
        st.session_state.generated.append(result["answer"]) 
    else:
        cit = ""
        st.session_state.generated.append("No results found, please rephrase the question")
   
    st.session_state.past.append(user_input)

if st.session_state['generated']:
    st.sidebar.subheader("References")
    st.sidebar.info(cit)
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], avatar_style="initials", seed="B", key=str(i))
        message(st.session_state['past'][i], is_user=True, avatar_style="initials", seed="A", key=str(i) + '_user')


