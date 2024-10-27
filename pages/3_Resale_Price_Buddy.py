from io import StringIO
import json
import os
import random
import requests
import time
from typing import Sequence

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
from typing_extensions import Annotated, TypedDict

from utility import check_password, check_openai_api

st.set_page_config(page_title="HDB Resale Price Buddy", page_icon="💸")
# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# endregion <--------- Streamlit Page Configuration --------->




st.title("💸HDB Resale Price Buddy")

"""
The HDB Resale Price Buddy is designed to help you query resale transaction price data for HDB flats in Singapore, based on data from 2017 onwards sourced from [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view?dataExplorer=). 

It can extract and understand key details from your queries, including specific towns, streets, blocks, flat types, storey ranges, and transaction months.
"""


with st.expander("How to use the app effectively:"):
    st.write("""

    - Ask about **specific streets** or **towns**: Example: "What are the resale prices for 3-room flats on Yishun Ring Road?"
    - Specify **block numbers** and **storey ranges** for targeted results: Example: "Show me prices for 4-room flats sold between the 1st and 9th floors at Block 123 Bishan Street 12."
    - Ask for historical prices by month and year: Example: "What were the average resale prices for Tampines in March 2020?"

    To get the best results, include as many details as possible about the flats you're interested in (street name, block number, town, flat type, storey range, or specific time periods).
    """
    )

headers = {'User-Agent': 
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

# Data Gov Web API Info
dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
url = "https://data.gov.sg/api/action/datastore_search?resource_id="  + dataset_id 

VALID_TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", 
    "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST", 
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", 
    "SENGKANG", "SERANGOON", "TAMPINES", "TOA PAYOH", "WOODLANDS", "YISHUN"
]

VALID_FLAT_TYPES = ["1 ROOM", "2 ROOM", "3 ROOM",  "4 ROOM", "5 ROOM", "MULTI-GENERATION", "EXECUTIVE"]

STREET_ABBREVIATIONS = {
    "north": "NTH",
    "south": "STH",
    "east": "EAST",
    "west": "WEST",
    "road": "RD",
    "avenue": "AVE",
    "street": "ST",
    "drive": "DR",
    "gardens": "GDNS",
    "crescent": "CRES",
    "central": "CTRL",
    "lorong": "LOR",
    "upper": "UPP",
    "bukit": "BT",
    "close": "CL",
    "heights": "HTS",
    "jalan": "JLN",
    "terrace": "TER",
    "place": "PL",
    "commonwealth": "C'WEALTH",
    
}

# Convert lists and abbreviations to string format for the prompt
valid_towns_str = ", ".join(VALID_TOWNS)
street_abbreviations_str = "; ".join([f"{k} → {v}" for k, v in STREET_ABBREVIATIONS.items()])
valid_flat_types_str = ", ".join(VALID_FLAT_TYPES)


@st.cache_resource
def init_llm(openai_api_key):
    llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key,  temperature=0)
    return llm

def get_entity_chain(llm):
    # Define a prompt template for extracting entities
    prompt_template = f"""
    You are an assistant helping a user query transaction price data from data.gov.sg about HDB resale flats. 
    A. Determine if the user query is related to HDB resale price information.
        If yes,
        - Determine from chat history {{chat_history_2}}  whether to reuse previously retrieved data or make a new API call
        - Extract necessary entities from each query, returning them in JSON format.

    If the question is about your data/knowledge source, kindly respond that your information is sourced from data.gov.sg and not from 
    any third-party or non-government websites.    

    Only if you are absolutely sure that the user's query is unrelated (e.g. "hello!", "do you like pancakes?", etc.), politely respond that you are not designed to converse about unrelated topics


    If you've determined that you need to make a new API call
    Extract the following entities based on the query context, returning them in JSON format:

    1. "street_name": The street name (convert any street name parts by using these (not case-sensitive) abbreviations: {street_abbreviations_str}). 
        For example, "Jalan bukit merah" -> "JLN BT MERAH"
        Extract this if a specific street is mentioned.

    2. "town": The town (must match one of these: {valid_towns_str}). 
        Extract this if no street name is provided but the query mentions a valid town.

    Additionally, if mentioned, extract the following optional entities:

    - "flat_type": The classification of flats by room size (must match one of these: {valid_flat_types_str}).
    - "block": A specific HDB block number, typically numerical (e.g., 182 or 182C). Often precedes the street name.
    - "storey_range": A floor range, expressed in intervals of 3 floors, are one of the following: "01 TO 03", "04 TO 06", "07 TO 09", ...
    - "month": A transaction month that must be in "YYYY-MM" format (e.g., "2023-10", "2018-07").

    Do not extract or add additional entities that are not mentioned above.

    ### Notes:
    - If multiple values for an entity are present, return them as a list.
    - If the query mentions a block or storey range not in the list of valid entities, ask the user for clarification.

    ### Examples:

    1. Query: "Return me resale prices for 3 and 4-room flats in 8B Upper Boon Keng Road"
        Extracted:
            "street_name": "UPP BOON KENG RD",
            "block": "8B",
            "flat_type": ["3 ROOM", "4 ROOM"]

    2. Query: "Return me prices for 1st to 9th floor 5-room flats in Yishun Ring Road"
        Extracted:
            "street_name": "YISHUN RING RD",
            "flat_type": "5 ROOM",
            "storey_range": ["01 TO 03", "04 TO 06", "07 TO 09"]

    3. Query: "Show 4-room flats sold in 180-185 Bedok North Road from October 2023 to April 2024"
        Extracted:
            "street_name": "BEDOK NTH RD",
            "block": ["180", "181", "182", "183", "184", "185"],
            "flat_type": "4 ROOM",
            "month": ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04"]

    4. Query: "Show 4 and 5-room flats in Marine Parade Central sold in January 2023 and March 2024"
        Extracted:
            "street_name": "MARINE PARADE CTRL",
            "flat_type": ["4 ROOM", "5 ROOM"],
            "month": ["2023-01", "2024-03"]

    5. Query: "What was the average price per HDB town in 2019?"
        Extracted:
            "month": ["2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06", "2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12" ]

    6. Query: "What are the average price of 5-Room flats at `Tampines Street 34` from Oct 2023 to Sep 2024, for each category of floor level?"
        Extracted:
            "street_name": "TAMPINES ST 34",
            "flat_type": ["5 ROOM"],
            "month": ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09"]         

    Question: {{input}}
    """

    # Initialize LangChain
    prompt_hdb = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder("chat_history_2"),
            ("human", "{input}"),
        ]
    )

    entity_extraction_chain = (
        RunnablePassthrough()
        | prompt_hdb
        | llm
        | JsonOutputParser()
    )                   

    return entity_extraction_chain


def run():
    ready = True
    
    # Check if the API key is in session_state
    api_key = check_openai_api()
    os.environ["OPENAI_API_KEY"] = api_key
    if api_key is None:
        ready = False

    if ready:

        llm = init_llm(api_key)
        entity_extraction_chain = get_entity_chain(llm)
        if "chat_history_2" not in st.session_state:
            st.session_state.chat_history_2=[
                AIMessage(content="Hi there! I'm the HDB Resale Price Buddy! What would you like to ask?")
            ]

        ### Statefully manage chat history ###
        class State(TypedDict):
            input: str
            chat_history_2: Annotated[Sequence[BaseMessage], add_messages]
            context: str
            answer: str

        class InputState(TypedDict):
            input: str
            chat_history_2: Annotated[Sequence[BaseMessage], add_messages]

        class EntityState(TypedDict):
            entities: str
        
        class DataState(TypedDict):
            data: str
            input: str

        class OutputState(TypedDict):
            chat_history_2: Annotated[Sequence[BaseMessage], add_messages]
            answer: str

        base_url="https://data.gov.sg/api/action/datastore_search"

        # Step 1: Define nodes in the graph
        def extract_entity(state: InputState):
            # # prompt = self.prompt_template.format(user_query=user_query)
            # extracted_entities = entity_extraction_chain.invoke(state)
            # st.write(extracted_entities)
            # return {
            #     "entities": extracted_entities
            # }
            try:
                # st.write(state)
                extracted_entities = entity_extraction_chain.invoke(state)
                if not extracted_entities:
                    return {
                        "entities": None,
                        "chat_history_2": [
                            HumanMessage(state["input"]),
                            AIMessage("Sorry, I couldn't extract any relevant information from your query."),
                        ],
                        "answer": "Sorry, I couldn't extract any relevant entities from your query."
                    }
                st.chat_message("AI").write(f"""
                Retrieving from data.gov.sg with the following parameters...

                {extracted_entities}
                    """)
                return {"entities": extracted_entities}
            # Future To-Do: Store dataframe state so I don't always have to remake API calls
            except Exception as e:
                return {
                    "entities": None,
                    "chat_history_2": [
                        HumanMessage(state["input"]),
                        AIMessage(f"An error occurred during entity extraction: {str(e)}"),
                    ],
                    "answer": f"Error: {str(e)}".replace('Error: Invalid json output: ','')
                }

        # Node 2: API Call Node as a Class
        def api_call(state: EntityState):
            if not state.get('entities'):
                return {} 
            url = f"{base_url}?resource_id={dataset_id}"
            api_params = {
                "filters": json.dumps(state['entities']),
                "limit": 10000,
                "offset": 0
            }

            all_dataframes = []
            total_retrieved = 0
            
            # Make as many API calls as needed to get all data
            # No restriction set for now (10000 rows per API call)
            while True:
                # Make the API call
                response = requests.get(url, params=api_params, headers=headers)
                data = response.json().get('result', {}).get('records', [])
                
                if not data:
                    break  # Exit loop if no more data is returned
                
                # Convert to DataFrame and append to the list
                df = pd.DataFrame(data)
                all_dataframes.append(df)
                
                # Update total rows retrieved
                total_retrieved += len(df)
                st.chat_message("AI").write(f"Retrieved {len(df)} rows (Total: {total_retrieved} rows)")
                
                # If less than 10,000 rows are returned, break the loop as it's the last page
                if len(df) < 10000:
                    break
                
                # Update offset for the next page
                api_params['offset'] += 10000
                time.sleep(round(random.random(), 2))

            # Make the API call here using requests as shown before
            # response = requests.get(url, params=api_params, headers=headers)
            # df = pd.DataFrame(response.json()['result']['records'])
            final_df = pd.concat(all_dataframes, ignore_index=True)
            st.chat_message("AI").write(f"""
                Retrieved the following data with total {total_retrieved} rows (up to first 100 shown)
                    """)
            st.write(final_df.head(100))
            return {
                "data": final_df.to_json()
            } 
            
        # Node 3: DataFrame Analysis Node as a Class
        def analyse(state: DataState):
            data = state['data']
            df = pd.read_json(StringIO(data))
            prefix = """
            You are a helpful assistant specialised in analysing data with a pandas dataframe in Python. The name of the dataframe is `df`.
            
            Provide only a concise natural language response with a polite tone that directly answers the question.

            If you are quoting a monetary value in your answer, add a '$' symbol before the value.
            
            Do not provide code. 

            """
            agent = create_pandas_dataframe_agent(llm,
                                                    df,
                                                    prefix=prefix,
                                                    allow_dangerous_code=True,
                                                    agent_type=AgentType.OPENAI_FUNCTIONS)
            response = agent.invoke(state['input'])
            #st.write(response)
            return {
                "chat_history_2": [
                    HumanMessage(state["input"]),
                    AIMessage(response["output"]),
                ],
                "answer": response["output"],
            } 

        # Define the workflow
        workflow = StateGraph(state_schema=State, input=InputState, output=OutputState)
        workflow.add_node("entity_extraction_node", extract_entity)
        workflow.add_node("api_call_node", api_call)
        workflow.add_node("dataframe_analysis_node", analyse)

        # Define the edges between nodes
        workflow.add_edge(START, "entity_extraction_node")
        # Conditional edge: only proceed if entities are found
        def should_call_api(state: EntityState):
            return "api_call_node" if state.get("entities") is not None else "END"
        # workflow.add_edge("entity_extraction_node", "api_call_node")
        workflow.add_conditional_edges("entity_extraction_node", 
                                    should_call_api,
                                    {
                                        "api_call_node": "api_call_node",
                                        "END": END
                                    })
        workflow.add_edge("api_call_node", "dataframe_analysis_node")

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    
        for message in st.session_state.chat_history_2:
            if isinstance(message,AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            else:
                with st.chat_message("Human"):
                    st.write(message.content)

        # User's prompt
        if prompt := st.chat_input():
            with st.chat_message("Human"):
                st.markdown(prompt)
            st.session_state.chat_history_2.append(HumanMessage(content=prompt))
        
            # Bot's response
            config = {"configurable": {"thread_id": "abc123"}}
            response = app.invoke(
                {"input": prompt,
                "chat_history_2": st.session_state.chat_history_2},
                config=config,
            )
            st.chat_message("AI").write(response["answer"].replace('$', '\$'))
            st.session_state.chat_history_2.append(AIMessage(content=response["answer"]))

        
    else:
        st.stop()
        
if __name__ == "__main__":
    run()