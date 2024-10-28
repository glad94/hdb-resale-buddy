# Common imports
import json
from random import randint
import requests
from time import sleep

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify as md
import pandas as pd
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

headers = {'User-Agent': 
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

# Define list of HDB web-urls for scraping
pages = [
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/overview",

    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/application-for-an-hdb-flat-eligibility-hfe-letter/income-guidelines",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/housing-loan-options",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/housing-loan-options/housing-loan-from-hdb",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/housing-loan-options/housing-loan-from-financial-institutions",
    
    "https://www.hdb.gov.sg/residential/buying-a-flat/working-out-your-flat-budget",
    "https://www.hdb.gov.sg/residential/buying-a-flat/working-out-your-flat-budget/ability-to-pay",
    "https://www.hdb.gov.sg/residential/buying-a-flat/working-out-your-flat-budget/budget-for-flat",
    "https://www.hdb.gov.sg/residential/buying-a-flat/working-out-your-flat-budget/credit-to-finance-a-flat-purchase",
    "https://www.hdb.gov.sg/residential/buying-a-flat/finding-a-flat",
    "https://www.hdb.gov.sg/residential/buying-a-flat/finding-a-flat/types-of-flats",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/finding-a-flat?anchor=resale-flat",
    "https://www.hdb.gov.sg/cs/infoweb/residential/buying-a-flat/finding-a-flat/resale-seminars",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/enhanced-cpf-housing-grant-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/cpf-housing-grants-for-resale-flats-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/step-up-cpf-housing-grant-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/couples-and-families/proximity-housing-grant-families",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/seniors",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/enhanced-cpf-housing-grant-singles",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/cpf-housing-grant-for-resale-flats-singles",
    "https://www.hdb.gov.sg/residential/buying-a-flat/understanding-your-eligibility-and-housing-loan-options/flat-and-grant-eligibility/singles/proximity-housing-grant-singles",

    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/mode-of-financing",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/planning-considerations",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/planning-considerations/managing-the-flat-purchase",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/planning-considerations/eip-spr-quota",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/planning-considerations/conversion-scheme-application-procedure",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/option-to-purchase",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/plan-source-and-contract/request-for-value",

    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/resale-application/application",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/resale-application/acceptance-and-approval",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/resale-application/request-for-enhanced-contra-facility",
    "https://www.hdb.gov.sg/residential/buying-a-flat/buying-procedure-for-resale-flats/resale-completion",
    "https://www.hdb.gov.sg/residential/buying-a-flat/conditions-after-buying",
]


def scrape_hdb_webdata(pages:list):
    """
    Scrapes HDB webpage data for the given list of pages.

    This function sends HTTP GET requests to each URL in the provided list of 
    HDB webpages, parses the HTML content to extract the main section, converts
    it to markdown format, and stores the URL and markdown text in a list.

    Args:
        pages (list): A list of URLs to scrape.

    Returns:
        list: A list of dictionaries, each containing 'page_url' and 'page_text'
            where 'page_text' is the markdown representation of the HTML content.
    """
    texts_information = []
    for page in tqdm(pages):
        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')
        page_md_str = md(str(pageSoup.find('section', class_="main-content")))
        
        texts_information.append({"page_url": page,
                            "page_text": page_md_str}) 
        
        sleep(randint(0,1))
    return texts_information


texts_information = scrape_hdb_webdata(pages)

with open('./data/data.json', 'w') as fp:
        json.dump(texts_information, fp)


# Create Vector Store
# Load data.json
with open('./data/data.json', 'rt') as f_in:
    docs_raw = json.load(f_in)
    
df_docs = pd.DataFrame(docs_raw)
df_docs = DataFrameLoader(df_docs, page_content_column="page_text")
list_docs = df_docs.load_and_split()

# Embedding model that we will use for the session
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

# Creating vector db from scratch
# Create the text splitter
text_splitter = SemanticChunker(embeddings_model)
splitted_documents = text_splitter.split_documents(list_docs)

# Embed each chunk and load it into the vector store.
vectordb = Chroma.from_documents(splitted_documents, embeddings_model, collection_name='embedding_semantic', persist_directory='./project/vector_db')