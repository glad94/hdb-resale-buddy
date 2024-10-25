import streamlit as st
from utility import check_password, check_openai_api

# region <--------- Streamlit Page Configuration --------->
st.set_page_config(
    page_title="HDB Resale Buddy",
    page_icon="🤖",
)

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# endregion <--------- Streamlit Page Configuration --------->

st.write("# HDB Resale Buddy 🤖")

# st.sidebar.success("Select a demo above.")
# Check if the API key is already stored in session_state
check_openai_api()

st.markdown(
    """
    ## Application Overview

    Welcome to the HDB Resale Buddy, your AI-Assistant for all your HDB resale buying enquiries. This application comprises two main features:

    ### Resale Q&A Buddy
    The Resale Q&A Buddy is designed to assist you with queries related to buying a resale HDB flat in Singapore. 
    
    Using data directly sourced from the [official HDB website](https://www.hdb.gov.sg/), it can answer questions about the resale procedure, financing options, eligibility, and other related topics. 
    
    For optimal results, provide specific and detailed queries related to procedures, rules, and official guidelines.

    ### Resale Price Buddy
    The Resale Price Buddy helps you query resale transaction price data for HDB flats in Singapore, based on data from 2017 onwards sourced from [data.gov.sg](https://data.gov.sg). 
    
    It can extract and understand key details from your queries, including specific towns, streets, blocks, flat types, storey ranges, and transaction months. 
    
    For best results, include as many details as possible about the flats you're interested in, such as street name, block number, town, flat type, storey range, or specific time periods.

    ---
    
    Explore these features by selecting a demo from the sidebar, and enjoy an insightful experience with the HDB Resale Buddy!
    """
)

with st.expander("DISCLAIMER"):
    st.write(
        """

        IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

        Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

        Always consult with qualified professionals for accurate and personalized advice.

        """
    )
