import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader
import transformers
from transformers import AutoConfig, AutoTokenizer
import torch
from torch import bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import re

def setup_retrievers():
    with open('/home/ebiz/SUSMITHA/3RDi/Medical/fintune_csv_xml/chatml_format_fintune/instanse/finetune_base_model/updated_corrected_db.json') as f:
        data = json.load(f)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device" : "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs= model_kwargs)
    chunks = []
    for json_obj in data:
        pd = json_obj['Patient Details']
        sec = json_obj['Section']
        ins = json_obj['Instruction']    
        page_content = f"Patient details: {pd}, section:{sec}, Instructions from doctor : {ins}"
        metadata = {'section' : sec}    
        document = Document(page_content=page_content, metadata=metadata)
        chunks.append(document)
        
    loader = DirectoryLoader('/home/ebiz/SUSMITHA/3RDi/updated_rag/pdf', glob="**/*.pdf", use_multithreading=True)       
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    doctor_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="Doctor_Corrected_DB")
    books_db = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="pdf_db")

    doctor_db = Chroma(persist_directory="./Doctor_Corrected_DB", embedding_function=embeddings)
    books_db = Chroma(persist_directory="./pdf_db", embedding_function=embeddings)

    doctor_db_client = doctor_db.as_retriever()
    books_db_client = books_db.as_retriever()

    return doctor_db_client, books_db_client

def setup_llm():
    model_name = "llama-3-8b-Instruct-bnb-4bit-updated_json"
    device = "cuda"

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        max_new_tokens=1024
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        torch_dtype=torch.float16,
        device_map=device,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=query_pipeline)
    return llm

def setup_retrieval_qa(doctor_db_client, books_db_client, llm):
    doctor_db_client_retriever = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=doctor_db_client, 
        verbose=True
    )

    books_db_client_retriever = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=books_db_client, 
        verbose=True
    )

    return doctor_db_client_retriever, books_db_client_retriever

def test_rag(qa, query):
    result = qa.run(query)
    return result

def get_corrected_text(query):
    doctor_db_client, books_db_client = setup_retrievers()
    llm = setup_llm()
    doctor_db_client_retriever, books_db_client_retriever = setup_retrieval_qa(doctor_db_client, books_db_client, llm)

    doc_query = f"""
    You're an assistant who provides only corrected text based on the knowledge you have. {query} Use this prompt to search for relevant information and provide the Corrected Text. If you don't find any relevant information, state that you have no knowledge on the topic.Give response within 200 words.
    """
    doctor_retriver = test_rag(doctor_db_client_retriever, doc_query)
    corrected_text_match = re.search(r"Helpful Answer:(.*)", doctor_retriver, re.DOTALL)
    corrected_text_doctor = corrected_text_match.group(1).strip()

    book_query = f"""
    You're an assistant who has book knowledge on UTIs.  If you have no knowledge on the topic, state that you have no knowledge on it.{query} Use this prompt to determine the type of UTI the patient has and recommend the appropriate medication. Analyze the patient's details.
    Provide segmented under the following headings:
    Diagnosis (What type of UTI)
    Treatement plan (Recommend Some medication according to their Diagnosis)
    """
    books_retriver = test_rag(books_db_client_retriever, book_query)
    corrected_text_match = re.search(r"Helpful Answer:(.*)", books_retriver, re.DOTALL)
    corrected_text_books = corrected_text_match.group(1).strip()

    return corrected_text_doctor, corrected_text_books
