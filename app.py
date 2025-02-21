import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Streamlit UI Config ---
st.set_page_config(page_title="LangChain Summarizer", page_icon="🦜", layout="wide")

# --- UI Enhancements ---
st.markdown("<h1 style='text-align: center;'>📜 AI Summarizer: YouTube & Websites</h1>", unsafe_allow_html=True)
st.markdown("#### 🤖 Powered by **LangChain & Groq** | Get quick summaries of web content! 🚀")
st.divider()

# Sidebar for API Key & Model Selection
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

    # Select Groq Model
    available_models = ["gemma2-9b-it", "llama3-8b", "mixtral-8x7b"]
    selected_model = st.selectbox("Select LLM Model", available_models)

st.markdown("### 🔗 **Enter a Website or YouTube Video URL**")
generic_url = st.text_input("Paste the URL here", label_visibility="collapsed", placeholder="https://...")

# --- Define System Message ---
system_message = """You are an advanced content summarizer. 
Your goal is to extract the most important insights from the given content while maintaining its original meaning.
Summarize the content concisely but ensure all key points, examples, and applications are included.
Use structured formatting with sections and bullet points.
"""

# --- Initialize LLM (Only if API Key is Provided) ---
if groq_api_key:
    llm = ChatGroq(model=selected_model, groq_api_key=groq_api_key)
else:
    st.error("❌ Please enter a valid Groq API Key in the sidebar before summarizing.")

# --- Summarization Prompts ---
map_prompt = PromptTemplate(template=system_message + "\n\nSummarize the following content in 400-700 words:\n{text}", input_variables=["text"])
combine_prompt = PromptTemplate(template=system_message + "\n\nCombine these summaries into one cohesive summary:\n{text}", input_variables=["text"])

if st.button("🔍 Summarize Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.warning("⚠️ Please provide a valid API key and URL!")
    elif not validators.url(generic_url):
        st.error("❌ Invalid URL! Please enter a valid website or YouTube link.")
    else:
        try:
            with st.spinner("🚀 Fetching & Processing... Please wait."):
                # --- Load Content ---
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                docs = loader.load()

                # --- Chunking for Long Content ---
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
                split_docs = text_splitter.split_documents(docs)

                # --- Summarization Chain ---
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )

                output_summary = chain.run(split_docs)
                word_count = len(output_summary.split())

                # --- Display Summary ---
                st.success("✅ Summary Generated!")
                st.markdown(f"### 📝 **Summary (Word Count: {word_count})**")
                st.info(output_summary)

                # --- Download Option ---
                st.download_button(
                    label="📥 Download Summary",
                    data=output_summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
