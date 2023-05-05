import streamlit as st
from langchain.llms import AzureOpenAI
from langchain.llms import OpenAI  
from langchain.llms import AzureOpenAI    

import os, json, re
import pandas as pd
import http.client
import sys
sys.path.append('./utils')

from serper import getSerperResults
from beautifulsoup import getPageContent

deployment_name = "davinci-003"
model_name = "text-davinci-text-003"
temperature = 0
max_tokens = 1000

past_events_dataset = "./data/event-history-dataset.csv"
daily_events_dataset = f"./data/ercc_daily_events-2023-04-12.json"

if "count" not in st.session_state:
    st.session_state.count = 0

@st.cache_resource
def getLLM(provider, model_name, temperature, max_tokens):
    if provider == "OpenAI":            
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_OPENAI_API_KEY"]
        os.environ["ORGANIZATION_KEY"] = st.secrets["OPENAI_ORGANIZATION_KEY"]            # key1 or key2        
        llm = OpenAI(model_name=model_name, temperature=temperature)
        return llm

    elif llm_provider == "Azure":      
        
        os.environ["OPENAI_API_TYPE"] = st.secrets["AZURE_OPENAI_API_TYPE"]         # azure
        os.environ["OPENAI_API_VERSION"] = st.secrets["AZURE_OPENAI_API_VERSION"]  # 2022-12-01
        os.environ["OPENAI_API_BASE"] = st.secrets["AZURE_OPENAI_API_BASE"]        # https://<name>.openai.azure.com
        os.environ["OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]          # key1 or key2
        deployment_name = "davinci-003"
        default_model_name = "text-davinci-text-003"
        # Create an instance of Azure OpenAI
        
        llm = AzureOpenAI(deployment_name=deployment_name, model_name=default_model_name, temperature=temperature, max_tokens=max_tokens)
        return llm

    else:
        return False

@st.cache_resource
def init(deployment_name, model_name, temperature, max_tokens, past_events_dataset, daily_events_dataset):
    llm = getLLM("OpenAI", "text-davinci-003", temperature, max_tokens)
    df_events = pd.read_csv(past_events_dataset, sep=";")
    df_daily_events = pd.read_json(daily_events_dataset)
    conn = http.client.HTTPSConnection("google.serper.dev")

    return llm, df_events, df_daily_events, conn

def getInternetResults(conn, event_details, latest_similar_events):
    for index, row in latest_similar_events.iterrows():
        url = row['source root url']
        query = f"{event_details} site:{url}"
        print(query)

        #Search the source web page for the event
        json_string = getSerperResults(conn, query, st.secrets["SERPER_API_KEY"] )
        json_data = json.loads(json_string)

        title = None
        link = None
        snippet = None
        date = None
        page_content = None

        if len(json_data['organic']) > 0:
            first_organic_result = json_data['organic'][0]
            title = first_organic_result['title']
            link = first_organic_result['link']
            snippet = first_organic_result['snippet']
            # date = first_organic_result['date']
            page_content, error = getPageContent(link)
    
        latest_similar_events.at[index, 'found title'] = title
        latest_similar_events.at[index, 'found link'] = link
        latest_similar_events.at[index, 'found snippet'] = snippet
        latest_similar_events.at[index, 'page content'] = page_content 
    return latest_similar_events

def count_words(text):
    words = re.findall(r'\w+', text)
    return len(words)

def get_middle_words(text, count):
    words = re.findall(r'\w+', text)
    start = (len(words) - count) // 2
    end = start + count
    return ' '.join(words[start:end])

def getEventReport(df_events, daily_event):
    event_name = daily_event['event_name']
    country_id = daily_event['country_id']
    event_id = daily_event['event_id']
    event_authority = daily_event['event_authority']

    # "Step 1: Query the past events to find specific Event ID and Country ID"    
    similar_events_df = df_events[(df_events['Event ID'] == event_id) & (df_events['Country ID'] == country_id) & (df_events['Link'] != 'NaN' )]
    similar_events_df = similar_events_df.dropna()
    # similar_events_df
    
    # "Step 2: Drop duplicate rows and add authority's urls"    
    for index, row in similar_events_df.iterrows():    
        try:    
            similar_events_df.at[index, 'source root url'] = row.Link.split("://")[0] + "://" + row.Link.split("://")[1].split("/")[0] 
        except:
            similar_events_df.at[index, 'source root url'] = False
    
    dedupe_similar_events = similar_events_df.drop_duplicates(subset=['Title', 'Location', 'Country ID', 'Event ID', 'source root url'], keep='first')
    len(dedupe_similar_events)
    latest_similar_events = dedupe_similar_events[:5].copy()
    
    "---"
    st.write(f"Processing **{event_name}**")
    st.write(f"Latest 5 events...")

    latest_similar_events
        
    # "Step 3: Search corresponding event on each authority page"
    event_details = f"{event_name} {event_authority} {st.session_state.timing}"
    latest_similar_events_results = getInternetResults(conn, event_details, latest_similar_events)
    latest_similar_events_valid_results = latest_similar_events_results.dropna(subset=['found link'])
    # latest_similar_events_valid_results

    # "Step 4: Get response from GPT using the text from the found authority pages"    
    context = ""
    sources = []
    for index, row in latest_similar_events_valid_results.iterrows():
        context += row['page content']
        sources.append(row['found link'])

    # temp workaround to avoid token limit : need to be replaced with embeddings
    word_count = count_words(context)
    if word_count > 1000:
        context = get_middle_words(context, 800)

    prompt_prefix = f"You will act as a journalist. Given the following [context], extract all available details concerning the {event_name}."
    prompt_suffix = f"Ideally from the following source: {event_authority}. You will not invent new facts."
    partial_prompt = f"{prompt_prefix} {prompt_suffix}"

    prompt  = f"{prompt_prefix}.  [context]: {context}. {prompt_suffix}"
    
    llm_answer = llm(f"{prompt}")    
    return llm_answer, partial_prompt, sources

# Init -------------------------------------------------

llm, df_events, df_daily_events, conn = init(deployment_name, model_name, temperature, max_tokens, past_events_dataset, daily_events_dataset)


# -------------------------------------------------
#                   Front end
# -------------------------------------------------

"""
# GPT for drafting ECHO Daily Flash
Your Efficient Solution for ERCC's ECHO Daily Flash Draft Reports

When natural disasters like floods and landslides strike, EventGuard simplifies preparing draft reports for the ECHO Daily Flash of the Emergency Response Coordination Centre (ERCC). Our AI-powered system:

1. Use the ERCC internal database to **Lookup for similar events for the same country** and **event type**
2. **Search on internet for each specific event on all the found authorities web sites** and collect the pages contents
3. **Use GPT to extract event information from the collected pages** and **draft the report for each event**.

**For official information**: check ERCC Emergency Response Coordination Centre [ECHO Daily Flash](https://erccportal.jrc.ec.europa.eu/#/echo-flash-items/latest) website.
"""
st.divider()
st.sidebar.write("**Preferences**")
st.sidebar.text_input("Today's date", "12 April 2023", key="timing")

"""
#### Draft ECHO Daily Flash
"""
df_daily_events

sumbit = st.button("Create draft ECHO Daily Flash report")

if sumbit:
    draft_reports = []
    for index, daily_event in df_daily_events.iterrows():        
        llm_answer, partial_prompt, sources = getEventReport(df_events, daily_event)

        st.divider()
        st.write(f"#### {daily_event['event_name']}")
        st.write(f"({daily_event['event_authority']})")
        st.write(f"{llm_answer}")
        st.write(f"Sources: {sources}")
        
        draft_reports.append({
            "event_name": daily_event['event_name'],
            "event_authority": daily_event['event_authority'],
            "event_id": daily_event['event_id'],
            "country_id": daily_event['country_id'],
            "timing": st.session_state.timing,
            "partial_prompt": partial_prompt,
            "sources": sources,
            "llm_description": llm_answer
        })
        
    st.session_state.draft_reports = draft_reports

