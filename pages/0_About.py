import streamlit as st
from utility import check_password, check_openai_api

st.set_page_config(page_title="About", page_icon="🏢")
st.title("🏢 About")

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
    This application was developed as part of the Project Phase of the Pilot Run of GovTech's **AI Champions**
    Bootcamp (2024), under Project Type C: Capstone Assignment.

    ### Objectives
    - Develop a web-based application that enables citizens to interact seamlessly with publicly available 
    information regarding a specific process or transaction with the government. 
    - The application should consolidate information or data from multiple official or trustworthy sources, 
    facilitate a deeper understanding through interactive engagements, and present the relevant information 
    in the way that is customised based user inputs and in the highly effective formats for the users to consume the information.
    
    ### Data Sources
    - Official HDB Website
    - HDB Resale Flat Transactions (2017 onwards) - Data.gov.sg

    Although data is sourced from official channels, it is not guaranteed that the AI assistants will 
    repond with 100% accurate and factual information. Please view the usage disclaimer on the Home page.
    
    """