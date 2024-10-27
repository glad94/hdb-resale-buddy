__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
from typing import Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from typing_extensions import Annotated, TypedDict

from utility import check_password, check_openai_api


st.set_page_config(page_title="HDB Resale Q&A Buddy ", page_icon="🏢")
st.title("🏢 HDB Resale Q&A Buddy ")

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# endregion <--------- Streamlit Page Configuration --------->


"""
    The Resale Q&A Buddy is designed to assist you with queries related to buying a resale HDB flat in Singapore. 
    
    Using data directly sourced from the [official HDB website](https://www.hdb.gov.sg/), the buddy can answer questions about the resale procedure, financing options, eligibility, and other related topics.
    **Note** As of now (28 Oct 2024), this app only has knowledge from only webpage content (excluding PDFs loaded through .aspx)
"""

with st.expander("How to use the app effectively:"):
    st.write(
    """

    - **Be specific:** Provide clear and detailed queries for better results.
    - **Ask process-related questions:** The app is optimised to provide information on procedures, rules, and official guidelines.
    - **Avoid unrelated questions:** This app focuses on HDB resale processes and official guidelines. Queries about topics outside this domain will not be answered.

    Try asking questions like:

    - "What documents do I need to complete the resale transaction?"
    - "What is the maximum financing I can get with my CPF for a resale flat?"
    - "How do I apply for an HFE letter?"

    For more accurate answers, please ensure your queries are within the app's scope!

    """
    )

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def get_metadata_source(docs):
#     return "\n\n".join([doc.metadata['page_url'] for doc in docs])

@st.cache_resource
def init_llm(openai_api_key):
    """
    Initialise embeddings, llm and vector store
    """
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
    # llm to be used in RAG pipeplines in this notebook
    llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key, temperature=0.1, streaming=True)

    # Load from stored/persisted vector db
    vectordb = Chroma(embedding_function=embeddings_model, collection_name='embedding_semantic', persist_directory="./vector_db")

    retriever = vectordb.as_retriever(search_type='mmr',
                                search_kwargs={'k': 3, 'fetch_k': 5}) # Use Maximum Marginal Relevance (MMR)

    return retriever, llm


def get_rag_chain(llm, retriever):

    ### Contextualize question ###
    """
    Create a Retrieval-Augmented Generation (RAG) chain that processes user questions by contextualizing them 
    with chat history and answering them using an LLM and a retriever.

    The function first reformulates the user question into a standalone question using a contextualizing prompt 
    to ensure it is understandable without previous chat history. It then creates a history-aware retriever 
    to manage retrieval of relevant context.

    After retrieving the relevant context, the function constructs a question-answering chain using a system 
    prompt tailored for question-answering tasks with constraints on source usage and metadata citation. 
    The RAG chain combines these components to deliver informed answers based on provided context.

    Args:
        llm: The language model used for generating responses.
        retriever: The retriever used for fetching contextual documents.

    Returns:
        A RAG chain that processes and answers user questions with historical context consideration.
    """
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history_1"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
    """You're a helpful AI assistant for question-answering tasks.
    You are provided with the following context information: {context}
    
    Answer the following question from the user.
    Use only information from the previous context. If the question requires mathematical calculations, think step-by-step and explain your answer.

    Each document in the context comes with metadata and page_content. The metadata contains important information
    such as the source URL. If the content from the document includes markdown or links, do not confuse these with the source URL. 
    The correct page URL for reference is provided in the metadata as "page_url" and must begin with https://www.hdb.gov.sg/. For example:

    Example 1:
        Document(metadata={{'page_url': 'https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/option-to-purchase'}}, page_content='Option to Purchase\n==================\n\n\n\n\nAfter you have obtained a valid [HDB Flat Eligibility (HFE) letter](/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter)')
        CORRECT metadata page_url: https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/option-to-purchase
        WRONG: /residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter
    
    Example 2:
        Document(metadata={{'page_url': 'https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/mode-of-financing'}}, page_content="| Remaining lease of flat is at least 20 years and can cover the youngest buyer up to the age of 95 |\n\n\nFor information on the use of CPF savings, you may use the CPF Board's [online calculator](https://www.cpf.gov.sg/member/tools-and-services/calculators/cpf-housing-usage)    
        CORRECT metadata page_url: https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/mode-of-financing
        WRONG: https://www.cpf.gov.sg/member/tools-and-services/calculators/cpf-housing-usage

    Explicitly list the metadata "page_url" used (these must not be pages outside of www.hdb.gov.sg/) at the end of your answer in the numbered-list:
    
    Sources:

    If the query is not related to the HDB resale process (e.g. What is the Prime Minister's Salary?, Help me understand tennis scoring?, etc.), politely tell 
    the user that you are not designed to converse about unrelated topics and request for the user to provide a query relating to your knowledge domain.

    You can repond politely and normally to casual conversation not specific to any other topic (e.g. Hi There! / How are you?)

    If the question is about your data/knowledge source, politely tell the user that your information is sourced from HDB's
    official website https://www.hdb.gov.sg/ and not from any third-party or non-government websites.

    If none of the contextual information can answer the topic-relevant question, just return the following message:
    Sorry, I'm not able to provide you with an answer to your question from my current knowledge base. You may wish
    to contact HDB via one of their channels at https://www.hdb.gov.sg/contact-us for further assistance.

    Question: {input}

    Answer: 
    """
    
    )
    
    ### From HDB BTO APP
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history_1"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Define the RAG chain with history handling
    # rag_chain = (
    #     RunnableWithMessageHistory(
    #         RunnableMap({
    #             "context": retriever | format_docs,
    #             "metadata": retriever | get_metadata_source,
    #             "query": RunnablePassthrough()
    #         }),
    #         lambda session_id: msgs,
    #     )
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    
    return rag_chain


def run():
    ready = True
    # Check if the API key is in session_state
    api_key = check_openai_api()
    os.environ["OPENAI_API_KEY"] = api_key
    if api_key is None:
        ready = False
        
    if ready:
        retriever, llm = init_llm(api_key)
        st.success("Knowledge base loaded!")

        # Step 2: Set up the RAG chain (with history awareness)
        rag_chain = get_rag_chain(llm=llm, retriever=retriever)

        # Step 3: Set up a text input field for user queries
        if "chat_history_1" not in st.session_state:
            st.session_state.chat_history_1=[
                AIMessage(content="Hi there, I'm the HDB Resale Q&A Buddy! What would you like to ask?")
            ]

        ### Statefully manage chat history ###
        class State(TypedDict):
            input: str
            chat_history_1: Annotated[Sequence[BaseMessage], add_messages]
            context: str
            answer: str


        def call_model(state: State):
            """
            Invoke the RAG chain and update the chat history.

            Args:
                state: The current state of the chat

            Returns:
                The updated state with the new chat history and the model's response
            """
            response = rag_chain.invoke(state)
            return {
                "chat_history_1": [
                    HumanMessage(state["input"]),
                    AIMessage(response["answer"]),
                ],
                "context": response["context"],
                "answer": response["answer"],
            }

        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        # Supposedly implementing memory but doesn't seem to register when checking
        # app.get_state(config).values["chat_history_1"] after each query
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        for message in st.session_state.chat_history_1:
            if isinstance(message,AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            else:
                with st.chat_message("Human"):
                    st.write(message.content)

        def stream_response(response):
            for word in response.split(" "):
                yield word + " "
                time.sleep(0.05)


        # User's prompt
        if prompt := st.chat_input():
            with st.chat_message("Human"):
                st.markdown(prompt)
            st.session_state.chat_history_1.append(HumanMessage(content=prompt))
        
            # Bot's response
            config = {"configurable": {"thread_id": "1"}}
            response = app.invoke(
                {"input": prompt,
                "chat_history_1": st.session_state.chat_history_1}, # Using chat history stored in Streamlit instead
                config=config,
            )
            # st.chat_message("AI").write(app.get_state(config).values["chat_history_1"])
            st.chat_message("AI").write_stream(stream_response(response["answer"].replace('$', '\$')))
            #st.chat_message("AI").write("\n\n".join([doc.metadata['page_url'] for doc in response["context"]])) 
            st.session_state.chat_history_1.append(AIMessage(content=response["answer"]))

        
    else:
        st.stop()
        
if __name__ == "__main__":
    run()