import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag import get_corrected_text

model_name = "llama-3-8b-Instruct-bnb-4bit-updated_json"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

def generate_response(doctor, books, prompt):
    messages = [
        {"role": "system", "content": f"""
        You are a professional doctor's assistant specializing in providing solutions for urinary tract infections (UTIs). Consider Doctor's corrections: {doctor} and the information retrieved from books: {books} and give response. Below are the details of a patient: {prompt}.
        Duty: Analyze the patient details, Doctor's corrections, and the information retrieved from books then provide:
        1. Diagnosis (What is the patient's problem, whether he/she has UTI or not? If UTI, then what kind of UTI it is?)
        2. Evidence (On what basis are you diagnosing him/her with that particular diagnosis?)
        3. Treatment (What type of medication does he/she need to undergo for that particular diagnosis?)
        4. Notes
        """},
        {"role": "user", "content": """Each section should be detailed and cover all relevant aspects. Aim for a comprehensive response of at least 512 words. Segments with headings and explaining them are very important. Use the following headings exactly:
        1. Diagnosis:

        2. Evidence:

        3. Treatment:

        4. Notes:"""},
    ]
   
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    res = tokenizer.decode(response, skip_special_tokens=True)
    
    return res

st.title("3RDi")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Call the get_corrected_text function to get responses from RAG
    doctor, books = get_corrected_text(prompt)
    
    response = generate_response(doctor, books, prompt)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
