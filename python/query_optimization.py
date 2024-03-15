import os
import time
import coloredlogs
import logging
import logging.config
import argparse
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_community.llms import OpenAI
from dotenv import load_dotenv

load_dotenv(".env", override=True)

#load_dotenv(r"C:\Users\hopewang\OneDrive - Microsoft\Desktop\FHL2024March\rag-query-optimization\python\workspace.env", override=True)
#### Execution example ####
# python query_optimization.py --directory ".\examplefolder"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

logger.debug(f"LANGCHAIN_ENDPOINT: {LANGCHAIN_ENDPOINT}")
 
def parse_arguments():
    """
    Parse command line arguments from a configuration file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='The directory path where your documentation is stored.')

    return parser.parse_args()

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def load_documents(directory):
    """
    Iteratively load documents from a directory and index them.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            docs = loader.load()
            documents.extend(docs)
    return documents
    
def load_and_split():
    # STEP 1: Load Documents from directory
    docs = load_documents(args.directory)
    logger.debug(f"Loaded {len(docs)} documents from {args.directory}")
    # STEP 2: Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, 
        chunk_overlap=50)

    splits = text_splitter.split_documents(docs)
    logger.debug(f"Split {len(docs)} documents into {len(splits)} splits")
    return splits

def vectorstore_indexing(splits):
    # STEP3: Vectorstore
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    logger.debug(f"Indexed {len(splits)} splits into vectorstore")
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_generation(question, retriever):
    # STEP 4: RAG
    docs = retriever.get_relevant_documents(question)

    rag_template = read_prompt('prompts/template_prompt.txt')
    prompt = ChatPromptTemplate.from_template(rag_template)
    chain = prompt | llm
    chain.invoke({"context":docs,"question":question})

def multi_query_generation(question, retriever):
    # STEP 4: Multi Query: Different Perspectives
    multi_query_template = read_prompt('prompts/multi_query_prompt.txt')
    logger.debug(f"multi_query_template is {multi_query_template}")
    prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

    #STEP 5: Generate Queries from Perspectives
    generate_queries = (
        prompt_perspectives 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    #STEP 6: Retrieve Documents
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})

    doc_retrival_template = read_prompt('prompts/template_prompt.txt')
    logger.debug(f"Retrieved {len(docs)} documents")
    prompt = ChatPromptTemplate.from_template(doc_retrival_template)

    final_rag_chain = (
        {"context": retrieval_chain, 
         "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain.invoke({"question":question})

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def rag_fusion_generation(question, retriever):
    template = read_prompt('prompts/rag_fusion_prompt.txt')
    logger.debug(f"read rag_fusion_prompt template: {template}")
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_rag_fusion 
        | ChatOpenAI(temperature=0)
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    # not used
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    logger.debug("retrieval_chain_rag_fusion is prepared")
    logger.debug(f"Retrieved {len(docs)} documents")

    template = read_prompt('prompts/template_prompt.txt')
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain.invoke({"question":question})

def rag_hyde_generation(question, retriever):
    template = read_prompt('prompts/hyde_prompt.txt')
    prompt_hyde = ChatPromptTemplate.from_template(template)
    logger.debug(f"read hyde_prompt template: {template}")
    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser() 
    )
    
    generate_docs_for_retrieval.invoke({"question":question})

    retrieval_chain = generate_docs_for_retrieval | retriever 
    retireved_docs = retrieval_chain.invoke({"question":question})
    retireved_docs

    template = read_prompt('prompts/template_prompt.txt')
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    final_rag_chain.invoke({"context":retireved_docs,"question":question})


def read_prompt(prompt_file_path):
    with open(prompt_file_path, 'r') as file:
        prompt = file.read().replace('\n', '')
    return prompt

def main(args):
    logger.debug("-----------------STARTING------------------------------")

    logger.debug("-----------------INDEXING------------------------------")
    splits = load_and_split()
    retriever = vectorstore_indexing(splits)
    
    logger.debug("-----------------RUNTIME------------------------------")
    # Set up timer to 1 hour to quit for the following while loop
    start_time = time.time()
    logger.debug("Timer started... Your application will quit after 1 hour.")

    second_question = read_prompt('prompts/second_question.txt')
    logger.debug(second_question)

    while True:
        if time.time() - start_time < 3600:
            pass
        else:
            logger.debug("Timer expired. Your application is quiting.")
            break
    
        question = input("Ask a question : ")
        if not question:
            continue
        logger.debug(f"Question: {question}")
        question2 = input(second_question)
        if not question2:
            logger.debug("Please select a valid option")
            continue
        if question2 == "1":
            logger.debug("You made an old-fasion choice to use RAG!")
            rag_generation(str(question), retriever)
        elif question2 == "2":
            logger.debug("You made a great choice to use RAG multi-query!")
            multi_query_generation(str(question), retriever)
        elif question2 == "3":
            logger.debug("You made a great choice to use RAG fusion!")
            rag_fusion_generation(str(question), retriever)
        elif question2 == "4":
            logger.debug("You made a great choice to use RAG Decomposition!")
            # TODO: Implement RAG Decomposition
        elif question2 == "5":
            logger.debug("You made a great choice to use RAG HyDE!")
            rag_hyde_generation(str(question), retriever)
        elif question2 == "6":
            logger.debug("You made a great choice to use RAG step back!")
            # TODO: Implement RAG step back
        else:
            logger.debug("Please select a valid option!")
logger.debug("-----------------ENDING------------------------------")

if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()
    logger.debug(f"Input directory is {args.directory}")
    main(args)