import os 

import streamlit as st
from streamlit_chat import message

from utils import get_query_from_question, load_api_keys, get_pubmed_articles, get_details, text_split

max_nr_results = 1

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

st.header("Streamlit Chat")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ","What are some treatments for diabetic macular edema?", key="input")
    return input_text 

load_api_keys()
openai_api_key = os.getenv('OPENAI_API_KEY')

user_input = get_text()

if user_input:
    output = get_query_from_question(user_input, 'gpt-3.5-turbo', openai_api_key)
    pm_ids = get_pubmed_articles(output,max_nr_results)
    details = get_details(pm_ids)
    #text_chunks = text_split(details['abstract'])

    st.success(output)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


