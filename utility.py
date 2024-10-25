import streamlit as st  
import random  
import hmac  

# """  
# This file contains the common components used in the Streamlit App.  
# This includes the sidebar, the title, the footer, and the password check.  
# """  

def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False

def check_openai_api():
    """  
    Checks if the OpenAI API Key is available in the session state, and if not, asks the user to input it.  
    If the user inputs an API key, it is stored in the session state.  
    If the API key is available, a success message is shown. Otherwise, an info message is shown asking the user to enter an OpenAI API Key.  
    If the user does not enter an API key, the app is stopped.  
    """
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = None

    # Check for OpenAI API Key in secrets or ask the user to input
    if "openai_api_key" in st.secrets:
        st.session_state["openai_api_key"] = st.secrets["openai_api_key"]
    else:
        api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
        if api_key_input:
            st.session_state["openai_api_key"] = api_key_input

    # If API key is available, show success, else ask to provide
    if st.session_state["openai_api_key"]:
        with st.sidebar:
            st.success("API Authenticated!")
        return st.session_state["openai_api_key"]
    else:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()
        return