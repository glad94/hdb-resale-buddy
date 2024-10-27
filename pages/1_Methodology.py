import streamlit as st
from utility import check_password, check_openai_api

st.set_page_config(page_title="Methodology", page_icon="🏢")
st.title("🏢 Methodology")

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# endregion <--------- Streamlit Page Configuration --------->

ready = True
# Check if the API key is in session_state
openai_api_key = check_openai_api()
if openai_api_key is None:
    ready = False
    
if ready:

    """
    Last Updated: 27 Oct 2024

    The Resale Q&A and Price Buddies are implemented through the Langchain and Langgraph libraries.


    ### Resale Q&A Buddy
    Resale Q&A Buddy is a Retrieval Augmented Generation (RAG) application that provides assistance with queries related to buying a resale HDB flat in Singapore.

    1. **Vector Store Creation**: The buddy's knowledge base is created "offline" by scraping pages from the official HDB website
    that relate to the resale buying process. The contents are then semantically chunked, embedded via the 'text-embedding-3-small'
    model, and stored in a vector store.

    ```python
    # Where df_docs is a dataframe with columns 'page_url' and 'page_text'
    DataFrameLoader(df_docs, page_content_column="page_text")

    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    # Create the text splitter
    text_splitter = SemanticChunker(embeddings_model)
    splitted_documents = text_splitter.split_documents(list_docs)

    # Embed each chunk and load it into the vector store.
    vectordb = Chroma.from_documents(splitted_documents, embeddings_model, collection_name='embedding_semantic', persist_directory='./project/vector_db')

    ```

    2. Upon password and API authentication, a Retriever is constructed from the above Vector Store using the "Maximal Marginal Relevance (MMR)" search type.
    The LLM to be used is also defined ('gpt-4o-mini') with temperature set at 0.1 for some minor variation.

    3. A RAG chain for question-answering is constructed via,
        - `history_aware_retriever` to reformulate the user question into a standalone question if it makes reference to any information in the chat history.
        - `create_stuff_documents_chain` to accept the retrieved context alongside the conversation history and query to generate an answer.
        - `create_retrieval_chain` which connects the above two to ensure that the responses incorporate both retrieved data and conversational history.

    4. The overall chat model is wrapped in a minimal LangGraph application to automatically persist the message history and faciliate the multi-turn application.
        (Note: this wasn't entirely successful so I ended up storing and using the chat history as part of Streamlit's session state.)
    
    ---

    ### Resale Price Buddy

    Resale Price Buddy is assistant for specific queries related to resale HDB flat prices, based on data from 2017 onwards sourced from [data.gov.sg](https://data.gov.sg).

    Langgraph is used again to create a sequential workflow that passes the user query through three nodes:

    1. **Entity Extraction:** If the user query is deemed relevant, extracts entities that relate to the HDB Resale Price dataset on data.gov.sg's API parameters.
        Some entity mapping dictionaries are provided to guide this node, e.g. mapping between all `street_name` components and their abbreviations ("Jalan" : "JLN").
        Extracted entities are passed on as a JSON string. For instance, 
        
        ```json
        {
        "street_name": "BEDOK NTH RD",
        "block": ["180", "181", "182", "183", "184", "185"],
        "flat_type": "4 ROOM",
        "month": ["2023-10", "2023-11", "2023-12"]
        }
        ```

        ***Not working entirely well:*** The assistant is fed with the chat history although its not always able to discern the correct entities to extract if given a follow-up prompt
        E.g. 
            - Query 1: Fetch me flat prices from block 8B Upper Boon Keng Road sold in 2022. 
            *Correct entities extracted*
            - Query 2: Fetch me for blocks 8A and 8C. 
            *May or may not discern that the query still refers to "Upper Boon Keng Road" and "2022".

    2. **API Call:** Simply takes in the API paramters and retrieves data from the source, returning a dataframe serialised into JSON.

    3. **Data Analysis:** Exploits the `create_pandas_dataframe_agent` agent to analyse the retrieved data according to the user query and returns the response. 
        
        ***Still buggy:*** I've noticed the agent gets caught "explaining the answer" rather than directly answering the query. Not sure if this seems to be mostly
        from more complex queries (e.g. calculate the annual trend), sample size is still small. This output is always truncated so I might need to try upping the 
        max tokens just to see. For now, I've tried to mitigate by adding a `prefix` that tells the LLM to provide the answer directly and not to show any code.

    
    """

else:
    st.stop()
