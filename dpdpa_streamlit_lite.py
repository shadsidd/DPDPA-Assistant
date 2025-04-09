import streamlit as st
from agno.agent import Agent
from agno.vectordb.chroma import ChromaDb
from agno.tools.duckduckgo import DuckDuckGoTools
#from agno.tools.thinking import ThinkingTools
#from agno.tools.knowledge import KnowledgeTools
import warnings
import time
import traceback
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "dpdpa_knowledge_lc_final_v5"
CHROMA_PERSIST_DIR = os.path.abspath("./dpdpa_chroma_lc_final_v5")
persist_path = Path(CHROMA_PERSIST_DIR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Clear the cache to force reinitialization
if 'vector_db' in st.session_state:
    del st.session_state['vector_db']
st.cache_resource.clear()

@st.cache_resource(show_spinner="Connecting to Knowledge Base...")
def initialize_vector_db():
    try:
        # Ensure directory exists
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        print(f"Attempting to connect to ChromaDB at: {CHROMA_PERSIST_DIR}")
        print(f"Collection name: {COLLECTION_NAME}")
        
        # Initialize ChromaDB with more explicit settings
        vector_db = ChromaDb(
            collection=COLLECTION_NAME,
            path=CHROMA_PERSIST_DIR,
            persistent_client=True
        )
        
        try:
            # Try to list all collections to verify connection
            all_collections = vector_db.client.list_collections()
            print(f"Available collections: {[col.name for col in all_collections]}")
            
            # Get or create collection
            collection = vector_db.client.get_or_create_collection(name=COLLECTION_NAME)
            collection_size = collection.count()
            print(f"Successfully connected to ChromaDB collection: {COLLECTION_NAME}")
            print(f"Collection size: {collection_size} documents")
            return vector_db
            
        except Exception as db_err:
            print(f"ChromaDB Error: {str(db_err)}")
            st.warning(f"Could not confirm connection to ChromaDB collection '{COLLECTION_NAME}'. KB search might fail. Error: {db_err}", icon="⚠️")
            return vector_db
            
    except Exception as e:
        st.error(f"Fatal error initializing vector database: {e}")
        print(f"Fatal error initializing vector database: {e}\n{traceback.format_exc()}")
        return None

@st.cache_resource(show_spinner="Waking up AI Assistants...")
def initialize_agents(_vector_db):
    if _vector_db is None:
        st.error("Cannot initialize agents: Vector database is not available.")
        return None, None

    try:
        dpdpa_agent_instructions_default = """You are an expert on India's Digital Personal Data Protection Act (DPDPA). Your goal is to explain complex topics in simple, easy-to-understand language for a general audience.
    
        For random questions let them know who you are and what you can assist them with.
**Response Guidelines (Default - Simple Mode):**

1.  **Start Simply:** Begin with a **clear, 1-sentence summary** directly answering the user's core question. Use simple terms.
2.  **Structure Clearly:** Organize the main explanation into logical sections using Markdown `###` headers (e.g., `### What the Law Says`, `### Key Requirements`, `### What This Means For You`). Choose headers relevant to the question. Use bullet points for lists. Limit to 2-4 key sections for simplicity.
3.  **Explain Jargon:** If you must use a legal or technical term, briefly explain it in plain English immediately after (e.g., "Data Fiduciary (this means the organization handling the data)..."). Keep explanations concise.
4.  **Focus on Practicality:** Emphasize the practical implications for individuals or organizations briefly.
5.  **"In Simple Terms" Summary:** **Conclude** the main explanation with a short section titled `### In Simple Terms:` summarizing the absolute key takeaways in 1-2 bullet points.
6.  **Handle "I Don't Know":** If the knowledge base lacks specific information: Clearly state that, provide general context *if possible*, and ALWAYS recommend concrete next steps (search online, check official text, consult legal expert).
7.  **Tone:** Be helpful, professional, reassuring, and concise.
"""

        dpdpa_agent_instructions_detailed_prefix = """**DETAILED MODE ACTIVATED:** Provide a comprehensive and highly detailed answer. Break the topic down into multiple specific sub-sections using relevant `###` Markdown headers. Explore nuances, specific regulations (citing sources if possible), potential challenges, and future outlooks where applicable. Aim for depth and thoroughness. **Do NOT include the 'In Simple Terms' final summary section.**

Based on the above, answer the following user query comprehensively:
"""

        dpdpa_agent = Agent(
            role="A helpful and clear DPDPA expert assistant",
            instructions=dpdpa_agent_instructions_default,
            #tools=[KnowledgeTools(knowledge=_vector_db, think=True, analyze=True, search=True)],
            search_knowledge=True,
            markdown=True,
            show_tool_calls=False
        )

        internet_agent = Agent(
            tools=[DuckDuckGoTools()],
            instructions="""You are an internet search assistant specializing in DPDPA. Find the absolute latest, relevant information on the user's query regarding DPDPA.
            
            For whatever user searches add DPDPA context to the search query if not there
**Output Guidelines:**

1.  **Summarize Findings:** Start with a brief summary of the key findings.
2.  **List Key Points:** Present detailed information as clear bullet points.
3.  **Cite Sources (URLs & Dates):** For each key point, include source URL and publication/access date if available. Format: "- [Point] (Source: [URL], [Date])".
4.  **Distinguish Fact vs. Report:** Differentiate official announcements from news/analysis.
5.  **Concise & Relevant:** try to  include  relevant information.
6.  **Disclaimer:** End with: "*Source: Recent Internet Search. Verify accuracy.*"
""",
            show_tool_calls=False,
            markdown=True,
        )
        print("Agents initialized successfully.")
        st.session_state.detailed_prefix = dpdpa_agent_instructions_detailed_prefix
        return dpdpa_agent, internet_agent
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        print(f"Error initializing agents: {e}\n{traceback.format_exc()}")
        return None, None

def format_response(response):
    answer = "Sorry, I encountered an issue processing the response."
    sources = []
    tool_calls = None
    try:
        if hasattr(response, 'content'):
            answer = response.content
            if hasattr(response, 'sources') and isinstance(response.sources, list):
                sources = response.sources
            tool_calls = getattr(response, 'tool_calls', None)
        elif isinstance(response, dict):
            for key in ['response', 'output', 'answer', 'text', 'content']:
                if key in response and isinstance(response[key], str):
                    answer = response[key]
                    break
            else: answer = str(response)
            if 'sources' in response and isinstance(response['sources'], list):
                 sources = response['sources']
            tool_calls = response.get("tool_calls", None)
        elif isinstance(response, str):
            answer = response
        else: answer = str(response)
    except Exception as e:
        print(f"Error formatting response: {e}\n{traceback.format_exc()}")
    return answer, sources, tool_calls

def format_sources(sources):
    if not sources: return ""
    formatted_list = []
    try:
        for i, source in enumerate(sources):
            source_info = []
            doc_name = getattr(source, 'document_name', None)
            page_label = getattr(source, 'page_label', None)
            url = getattr(source, 'url', None)
            if isinstance(source, dict):
                doc_name = source.get('document_name', doc_name)
                page_label = source.get('page_label', page_label)
                url = source.get('url', url)

            display_str = ""
            if doc_name: display_str += f"Doc: `{doc_name}`"
            if page_label: display_str += f" (Page: {page_label})"
            if url and isinstance(url, str) and url.startswith('http'):
                 display_url = url if len(url) < 70 else url[:67] + "..."
                 display_str += f" - <{url}|{display_url}>" if display_str else f"<{url}|{display_url}>"
            elif url: display_str += f" - `{url}`" if display_str else f"`{url}`"

            if display_str: formatted_list.append(f"   {i+1}. {display_str}")
            elif isinstance(source, str): formatted_list.append(f"   {i+1}. {source}")
            else: formatted_list.append(f"   {i+1}. `{str(source)[:100]}...`")

        return "\n\n---\n**Sources Consulted:**\n" + "\n".join(formatted_list) if formatted_list else ""
    except Exception as e:
        print(f"Error formatting sources: {e}\n{traceback.format_exc()}")
        return "\n\n---\n_(Error displaying sources)_"

def should_offer_internet_search(answer, sources):
    if not answer: return True
    if not sources: return True
    if len(answer.split()) < 30: return True
    keywords = ["unable to find", "no specific information", "don't have details", "recommend checking online", "could not find", "general understanding", "latest update"]
    if any(keyword in answer.lower() for keyword in keywords): return True
    return False

st.set_page_config(page_title="DPDPA Assistant", layout="wide", initial_sidebar_state="expanded")
st.title("🇮🇳 DPDPA Q&A Assistant")
st.caption("Ask about India's Digital Personal Data Protection Act. Get simple explanations or detailed breakdowns.")

vector_db = initialize_vector_db()
if not vector_db:
    st.error("Application initialization failed: Cannot connect to Vector DB.")
    st.stop()

dpdpa_agent, internet_agent = initialize_agents(vector_db)
if not dpdpa_agent or not internet_agent:
    st.error("Application initialization failed: Could not initialize AI agents.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_internet_search_prompt" not in st.session_state:
    st.session_state.pending_internet_search_prompt = None
if "show_detailed_answer" not in st.session_state:
    st.session_state.show_detailed_answer = False
if "detailed_prefix" not in st.session_state:
    st.session_state.detailed_prefix = """**DETAILED MODE ACTIVATED:** Provide a comprehensive and highly detailed answer. Break the topic down into multiple specific sub-sections using relevant `###` Markdown headers. Explore nuances, specific regulations (citing sources if possible), potential challenges, and future outlooks where applicable. Aim for depth and thoroughness. **Do NOT include the 'In Simple Terms' final summary section.**

Based on the above, answer the following user query comprehensively:
"""

with st.sidebar:
    st.info("**Your DPDPA Guide**\nGet quick summaries or deep dives into the DPDPA.", icon="💡")
    st.toggle("Show Detailed Answer", value=st.session_state.show_detailed_answer, key="show_detailed_answer", help="Get more comprehensive, multi-section answers.")
    st.markdown("---")
    if st.button("Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_internet_search_prompt = None
        st.rerun()
    st.markdown("---")
    st.caption("Always verify critical legal information.")

for idx, message in enumerate(st.session_state.messages):
    avatar = message.get("avatar", "🧑‍💻" if message["role"] == "user" else "🤖")
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("details"):
            details = message["details"]
            offer_search = details.get("offer_search", False)
            time_elapsed = details.get("time", 0)
            tool_calls = details.get("tool_calls")
            original_prompt = details.get("original_prompt")

            if avatar != "🌐":
                action_cols = st.columns([0.7, 0.3])
                with action_cols[0]:
                    if offer_search:
                        button_key = f"search_internet_{details['message_index']}"
                        if st.button("🌐 Search Internet for Latest", key=button_key, help="Check online for the most recent information.", type="primary"):
                            st.session_state.pending_internet_search_prompt = original_prompt
                            st.rerun()
                with action_cols[1]:
                    with st.expander("Technical Details"):
                        st.caption(f"KB Search Time: {time_elapsed:.2f}s")
                        st.write("**Tool Calls (KB Agent):**")
                        if isinstance(tool_calls, (dict, list)) and tool_calls:
                             st.json(tool_calls, expanded=False)
                        else:
                             st.write(tool_calls if tool_calls is not None else "N/A")

if st.session_state.pending_internet_search_prompt:
    search_prompt = st.session_state.pending_internet_search_prompt
    st.session_state.pending_internet_search_prompt = None
    print(f"--- Performing internet search for: {search_prompt} ---")
    try:
        start_internet_time = time.time()
        internet_message_placeholder = st.chat_message("assistant", avatar="🌐").empty()
        with internet_message_placeholder:
            with st.spinner("Searching the web for the latest DPDPA info..."):
                internet_response = internet_agent.run(f"Latest DPDPA info on: {search_prompt}")
                internet_answer, _, internet_tool_calls = format_response(internet_response)
                internet_result_content = f"""
> **Your Question:** {search_prompt}

**Recent Web Findings:**

{internet_answer}

---
*Source: Internet Search via DuckDuckGo. Please verify accuracy.*
"""
                st.markdown(internet_result_content)

        internet_time_elapsed = time.time() - start_internet_time
        st.session_state.messages.append({
            "role": "assistant",
            "avatar": "🌐",
            "content": internet_result_content,
            "details": {
                "time": internet_time_elapsed,
                "tool_calls": internet_tool_calls,
                "original_prompt": search_prompt
            }
         })
        st.rerun()
    except Exception as e_internet:
        st.error(f"Internet search failed: {e_internet}")
        st.session_state.messages.append({"role": "assistant", "avatar": "⚠️", "content": f"Sorry, the internet search failed: {e_internet}"})
        print(f"Internet search error: {e_internet}\n{traceback.format_exc()}")
        st.rerun()

if prompt := st.chat_input("Ask your question about DPDPA..."):
    if not dpdpa_agent or not internet_agent:
        st.error("Agents not available. Please check initialization.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍💻"})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    final_prompt_to_agent = prompt
    if st.session_state.show_detailed_answer:
        detailed_prefix = st.session_state.get("detailed_prefix", "")
        final_prompt_to_agent = detailed_prefix + prompt
        print("--- Running in DETAILED mode ---")
    else:
        print("--- Running in SIMPLE mode ---")

    try:
        start_time = time.time()
        assistant_message_placeholder = st.chat_message("assistant", avatar="🧠").empty()
        with assistant_message_placeholder:
            spinner_message = "Consulting Knowledge Base (Detailed Mode)..." if st.session_state.show_detailed_answer else "Consulting Knowledge Base..."
            with st.spinner(spinner_message):
                response = dpdpa_agent.run(final_prompt_to_agent)
                answer, sources, tool_calls = format_response(response)
                kb_response_content = answer
                kb_response_content += format_sources(sources)
                st.markdown(f"> **Your Question:** {prompt}\n\n{kb_response_content}")

            kb_time_elapsed = time.time() - start_time
            offer_search = should_offer_internet_search(answer, sources)
            current_message_index = len(st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "avatar": "🧠",
                "content": kb_response_content,
                "details": {
                    "offer_search": offer_search,
                    "time": kb_time_elapsed,
                    "tool_calls": tool_calls,
                    "original_prompt": prompt,
                    "message_index": current_message_index
                }
            })
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred processing your request: {str(e)}")
        error_message = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "avatar": "⚠️", "content": error_message})
        print(f"Error processing query '{prompt}': {e}\n{traceback.format_exc()}")
        st.rerun()

st.markdown("---")
st.caption("Disclaimer: This AI assistant provides informational summaries based on available data. It is not a substitute for professional legal advice. Always consult with a qualified legal expert for specific DPDPA compliance guidance.")