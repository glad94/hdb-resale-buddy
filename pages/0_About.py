import streamlit as st
from utility import check_password, check_openai_api

st.set_page_config(page_title="About", page_icon="❓")

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# endregion <--------- Streamlit Page Configuration --------->
st.title("❓ About")

ready = True
# Check if the API key is in session_state
openai_api_key = check_openai_api()
if openai_api_key is None:
    ready = False
    
if ready:

    """
    This application was developed as part of the Project Phase of the Pilot Run of GovTech's **AI Champions**
    Bootcamp (2024), under Project Type C: Capstone Assignment.

    Project Repository: https://github.com/glad94/hdb-resale-buddy/

    ### Objectives
    - Develop a web-based application that enables citizens to interact seamlessly with publicly available 
    information regarding a specific process or transaction with the government. 
    - The application should consolidate information or data from multiple official or trustworthy sources, 
    facilitate a deeper understanding through interactive engagements, and present the relevant information 
    in the way that is customised based user inputs and in the highly effective formats for the users to consume the information.
    
    ### Data Sources
    - [Official HDB Website](https://www.hdb.gov.sg/)
    - [HDB Resale Flat Transactions (2017 onwards) - Data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view?dataExplorerPage=19199)

    Although data is sourced from official channels, it is not guaranteed that the AI assistants will 
    repond with 100% accurate and factual information. Please view the usage disclaimer on the Home page.
    
    Feel free to get in touch with me through a channel on my [Github profile](https://github.com/glad94) :)
    """