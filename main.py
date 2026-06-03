import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# Modern stable LangChain primitives
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Groq and Core Runnable structures
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# check if url points to any file 
def is_valid_file_url(url):
    """Checks if the link points to a PDF, Word Doc, or Excel sheet."""
    extensions = ('.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv')
    return any(url.lower().endswith(ext) for ext in extensions)

#scrape and extract the data
def scrape_and_extract(url, download_dir="./downloads"):
    """
    Scrapes the base website text and automatically extracts data 
    from any embedded PDF, Docx, or Excel sheets.
    """
    os.makedirs(download_dir, exist_ok=True)
    documents = []
    print(f"Fetching webpage content: {url}")

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        plain_text = soup.get_text(separator="\n", strip=True)
        
        documents.append(Document(
            page_content=plain_text, 
            metadata={"source": url, "type": "webpage_text"}
        ))
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            if is_valid_file_url(full_url):
                file_name = os.path.basename(urlparse(full_url).path)
                local_filepath = os.path.join(download_dir, file_name)
                
                print(f"Downloading asset: {file_name}")
                try:
                    file_data = requests.get(full_url, timeout=15).content
                    with open(local_filepath, 'wb') as f:
                        f.write(file_data)
                    
                    # Unstructured dynamically handles PDFs, Word, Excel
                    loader = UnstructuredFileLoader(local_filepath, mode="single")
                    loaded_docs = loader.load()
                    
                    for doc in loaded_docs:
                        doc.metadata = {"source": full_url, "type": file_name.split('.')[-1]}
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"⚠️ Failed to process file {file_name}: {e}")
    except Exception as e:
        print(f"Failed to reach base URL {url}: {e}")    
    return documents    

#text split, embeddings and retriever
def build_vector_store(documents):
    """Splits all extracted contents into chunks and vectors them into Chroma."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split text & assets into {len(chunks)} contextual chunks.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 8,
            'fetch_k': 16, 
            'lambda_mult': 0.4
        }
    )

retriever_instance = None

@tool
def query_website_knowledge_base(query: str) -> str:
    """Use this tool to search the dynamic knowledge base compiled from the target website and files."""
    global retriever_instance
    if not retriever_instance:
        return "Error: No website data has been compiled yet."
    
    docs = retriever_instance.invoke(query)
    return "\n\n".join([f"[Source: {d.metadata['source']}]: {d.page_content}" for d in docs])



# main runnables 
def main_chat_session(user_url, user_question):
    global retriever_instance
    
    raw_docs = scrape_and_extract(user_url)
    if not raw_docs or (len(raw_docs) == 1 and raw_docs[0].page_content == ""):
        return "Failed to extract text data from the provided URL."
        
    retriever_instance = build_vector_store(raw_docs)
    print("Knowledge base built and vectorized successfully!")
    
    # 2. Configure Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.05,
        max_retries=3
    )
    
    llm_with_tools = llm.bind_tools([query_website_knowledge_base])
    
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an advanced AI assistant. Your task is to look at the user's question "
                   "and determine if you need to look up information from the scraped website "
                   "knowledge base to provide an accurate answer. If the question requires "
                   "reading or analyzing the website data, you MUST call the query_website_knowledge_base tool."),
        ("human", "{input}")
    ])
    
    # Reusable Final Template matching your style perfectly
    final_prompt = ChatPromptTemplate.from_template("""
You are an advanced AI Data Specialist and Knowledge Assistant. 

GOAL: 
Provide a precise, comprehensive, and objective response to the user's question using ONLY the provided website Context. 

INSTRUCTIONS:
1. **Scope**: If the user asks a general question about the website's offerings, services, or "what they do," synthesize all main pillars or sections found within the provided context.
2. **Specificity**: If the user asks a specific question (e.g., details about a project, report, data figures, or pricing), focus deeply on that data while pulling explicit details from the context.
3. **Tone**: Professional, objective, informative, and direct. Avoid generic filler words or assumptions not supported by the data.
4. **Formatting**: Use a clear bulleted list when presenting multiple items or points. **Bold** key terms, metrics, specific documents, or vital statistics for scannability.
5. **Fallback**: If the specific information requested cannot be found anywhere within the provided context, state clearly: "I am sorry, but the provided website data does not contain specific details on that topic." Do not invent facts.
6. **Citations**: Whenever possible, subtly mention the source file name or section title where you pulled the facts from, so the user knows exactly where it lives.

Context:
{context}

Question: {question}
""")
    
    chain = routing_prompt | llm_with_tools
    
    print("Consulting llm")
    ai_msg = chain.invoke({"input": user_question})
    
    if ai_msg.tool_calls:
        print("Tool triggered. Fetching matching context fragments...")
        tool_call = ai_msg.tool_calls[0]
        search_query = tool_call["args"].get("query", user_question)
        
        tool_output = query_website_knowledge_base.invoke(search_query)
        
        final_chain = final_prompt | llm
        final_response = final_chain.invoke({"context": tool_output, "question": user_question})
        return final_response.content
        
    print("⚠️ Tool not triggered automatically. Pulling context forcefully via raw fallback query...")
    forced_context = query_website_knowledge_base.invoke(user_question)
    
    final_chain = final_prompt | llm
    final_response = final_chain.invoke({"context": forced_context, "question": user_question})
    return final_response.content


if __name__ == "__main__":
    test_url = "https://www.gov.wales/sustainable-communities-learning-business-case-guidance"
    
    question = "What specific templates or tools are available to download on this page?"
    
    print("\n--- 🚀 Starting Scraper Agent Test Session ---")
    answer = main_chat_session(test_url, question)
    
    print("\n🤖 Final Answer From Groq Bot:")
    print(answer)