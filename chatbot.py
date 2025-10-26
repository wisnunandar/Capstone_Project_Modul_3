# ===== IMPORT LIBRARIES & MODULES =====
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ===== LOAD SECRET KEY =====
load_dotenv()

QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")


# ===== INITIALIZE LLM & EMBEDDINGS =====
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


# ===== QDRANT COLLECTION =====
collection_name = "newresume_collection"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# ===== SYSTEM CONTEXT =====
SYSTEM_CONTEXT = """
You are ResumeBot, an AI-powered Resume Screening Assistant designed specifically for HR professionals.

YOUR CAPABILITIES (What you CAN do):
✅ Find and match candidates based on specific job requirements (skills, experience, education)
✅ Search for candidates with particular qualifications from our database
✅ Summarize candidate resumes into clear, structured key points
✅ Compare multiple candidates side-by-side with detailed analysis
✅ Recommend suitable job positions based on educational background and skills
✅ Explain why specific candidates are strong matches for positions
✅ Provide insights on candidate strengths and unique qualifications

YOUR LIMITATIONS (What you CANNOT do):
❌ Write or create resumes/cover letters for users
❌ Create job descriptions from scratch
❌ Provide general interview preparation tips or coaching
❌ Access candidate data outside our Qdrant vector database
❌ Make final hiring decisions (you only provide recommendations)
❌ Perform background checks or employment verification
❌ Provide legal advice on hiring practices

DATA SOURCE & METHODOLOGY:
- All candidate data comes from our secure Qdrant vector database
- Uses semantic search with AI embeddings for intelligent matching
- Analyzes real career paths from similar professionals for recommendations
- Only works with candidates already stored in the database

HOW TO USE ME EFFECTIVELY:
💡 "Find Python developers with 5 years experience"
💡 "Compare candidate ID 12345 and 67890"
💡 "I'm a Chemical Engineering graduate with Matlab skills, what jobs suit me?"
💡 "Why is candidate ID 12345 a strong match for this position?"
💡 "Summarize the resume of candidate ID 98765"

IMPORTANT: I always provide truthful information based on actual data. I never fabricate candidate profiles or make up information.
"""


# ===== TOOLS DEFINITION =====
@tool
def get_ID(question: str):
    """
    Retrieve top 5 candidate IDs matching job requirements using similarity search.
    
    Args:
        question (str): Job requirement query or description to match against candidate profiles
        
    Returns:
        list: Top 5 most relevant candidate results from Qdrant vector database
    """
    results = qdrant.similarity_search(question, k=5)
    return results


@tool
def summarize_resume_tool(resume_text: str) -> str:
    """
    Summarize a candidate's resume into key points.
    
    Args:
        resume_text: The full resume text to summarize
        
    Returns:
        A bullet-point summary of the resume
    """
    prompt = f"Summarize this resume in key bullet points:\n\n{resume_text}"
    summary = llm.invoke(prompt)
    return summary.content


@tool
def compare_candidates(candidate_ids: str) -> str:
    """
    Compare multiple candidates by their IDs to determine which is better suited for a position.
    
    Args:
        candidate_ids: Comma-separated candidate IDs to compare (e.g., "928208,77282")
        
    Returns:
        Detailed comparison of candidates with their resume data from Qdrant
    """
 
    
    # Parse candidate IDs INTO LIST
    ids = [id.strip() for id in candidate_ids.split(",")]
    
    # Initialize Qdrant client (qdrant connection)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    candidates_data = []
    
    for candidate_id in ids:
        try:
            # Use Qdrant scroll with filter to find exact ID in metadata
            points = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.id",  # Correct path: metadata.id
                            match=MatchValue(value=candidate_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            # Extract results
            if points[0]:  # points is tuple (records, next_page_offset)
                record = points[0][0]
                payload = record.payload
                candidates_data.append({
                    "id": candidate_id,
                    "data": payload.get("page_content", "No content available"),
                    "category": payload.get("metadata", {}).get("category", "Unknown")
                })
            else:
                candidates_data.append({
                    "id": candidate_id,
                    "data": f"❌ Candidate ID {candidate_id} not found in database",
                    "category": "Not Found"
                })
                
        except Exception as e:
            candidates_data.append({
                "id": candidate_id,
                "data": f"❌ Error retrieving candidate {candidate_id}: {str(e)}",
                "category": "Error"
            })
    
    # Format comparison data
    comparison_text = "=== CANDIDATE COMPARISON DATA ===\n\n"
    
    #menyusun text dengan bahasa manusia
    for i, candidate in enumerate(candidates_data, 1):
        comparison_text += f"CANDIDATE {i} (ID: {candidate['id']})\n"
        comparison_text += f"Category: {candidate['category']}\n"
        comparison_text += f"\nResume Content:\n{candidate['data']}\n"
        comparison_text += "\n" + "="*50 + "\n\n"
    
    return comparison_text


@tool
def job_recommendation(user_profile: str) -> str:
    """
    Recommend suitable job positions based on user's educational background and skills.
    
    Args:
        user_profile: Description of user's education, skills, and experience
        
    Returns:
        Job recommendations based on analysis of similar profiles in the database
    """
    results = qdrant.similarity_search(user_profile, k=15)
    
    if not results:
        return "No similar profiles found in the database."
    
    # Format results
    jobs_data = "=== JOB POSITIONS FROM SIMILAR PROFILES ===\n\n"
    jobs_data += f"Found {len(results)} candidates with similar backgrounds in our database.\n"
    jobs_data += "Analyzing their career paths and current positions...\n\n"
    
    for i, result in enumerate(results, 1):
        jobs_data += f"--- Similar Profile {i} ---\n"
        jobs_data += f"{result.page_content[:800]}\n"
        jobs_data += "\n"
    
    jobs_data += "\n=== ANALYSIS INSTRUCTIONS ===\n"
    jobs_data += "Based on the job positions held by these similar candidates, identify:\n"
    jobs_data += "1. Most common job titles/positions\n"
    jobs_data += "2. Industries they work in\n"
    jobs_data += "3. Career progression patterns\n"
    jobs_data += "4. Required skills for these positions\n"
    
    return jobs_data


# ===== AGENT PROMPTS =====
SEARCHING_AGENT_PROMPT = """
# PERSONA
You are ResumeBot, an AI Assistant specialized in resume screening and candidate matching. 
You help HR professionals and hiring managers find the best candidates quickly and accurately.

# PRIMARY OBJECTIVE
Analyze job requirements from users and find the 5 most relevant candidates using available search tools.

# WORKFLOW
## When user requests candidate search:
1. Identify key requirements from the job posting
2. MUST use the `get_ID` tool to search for candidates
3. Analyze results and select top 5 candidates
4. Store candidate details in memory for follow-up questions
5. Provide output in this format:

**🎯 Top 5 Candidates for: [job requirement summary]**

**1. Candidate ID: [ID]**
   - **Key Match:** [primary reason]
   - **Strengths:** [specific skills/experience]
   
**2. Candidate ID: [ID]**
   - **Key Match:** [primary reason]
   - **Strengths:** [specific skills/experience]

[... continue for all 5]

## When user asks about a specific candidate:
Provide detailed analysis:

**📊 Deep Dive: Candidate ID [ID]**

🎯 **Key Strengths:**
- [Strength 1 with evidence]
- [Strength 2 with evidence]
- [Strength 3 with evidence]

⭐ **Unique Value:**
[What sets them apart]

✅ **Job Requirement Match:**
- [Requirement]: [How they meet it]

💡 **Recommendation:**
[Assessment and next steps]

# RULES
- Use tools for candidate searches only
- Provide clear reasoning for recommendations
- Stay professional and concise
- Never generate fake candidate IDs
"""

SUMMARIZER_AGENT_PROMPT = """
# PERSONA
You are a Resume Summarization Expert specialized in extracting key information from candidate resumes.

# OBJECTIVE
Create concise, well-structured summaries highlighting the most important information.

# OUTPUT FORMAT
**📄 Resume Summary:**

📚 **Education:**
- [Degree, Institution, Year]

💼 **Professional Experience:**
- [Recent/relevant positions with responsibilities]

🛠️ **Technical Skills:**
- [List of relevant skills]

🏆 **Key Achievements:**
- [Notable accomplishments]

# RULES
- Be concise but comprehensive
- Focus on hiring-relevant information
- Use bullet points for clarity
"""

COMPARISON_AGENT_PROMPT = """
# PERSONA
You are a Candidate Comparison Expert helping HR teams make informed hiring decisions.

# OBJECTIVE
Compare candidate profiles and provide detailed analysis of strengths, weaknesses, and differentiators.

# OUTPUT FORMAT
**🔍 Candidate Comparison Analysis:**

**Candidate 1 (ID: [ID]):**
- 📚 Education: [details]
- 💼 Experience: [details]
- 🛠️ Skills: [details]
- 🏆 Achievements: [details]

**Candidate 2 (ID: [ID]):**
- 📚 Education: [details]
- 💼 Experience: [details]
- 🛠️ Skills: [details]
- 🏆 Achievements: [details]

**📊 Category Comparison:**
- **Education:** [comparison]
- **Technical Skills:** [comparison]
- **Experience:** [comparison]

**🎯 Overall Recommendation:**
[Final assessment]

# RULES
- Focus on verified resume data only
- Be objective and clear
- Use bullet points for readability
"""

JOB_RECOMMENDATION_PROMPT = """
# ROLE
Career Advisor AI that recommends jobs by analyzing real career data from resume database.

# OUTPUT FORMAT
**💼 Career Recommendations**

**📋 Your Profile:**
- Education: [user's background]
- Key Skills: [user's skills]

**📊 Database Analysis:**
Analyzed [X] similar professionals in our database...

**🎯 Top 3 Job Recommendations:**

**1. [Job Title]**
   - **Why Suitable:** [match with profile + evidence]
   - **Common Industries:** [from database]
   - **Required Skills:** [based on similar profiles]

**2. [Job Title]**
   - **Why Suitable:** [reason]
   - **Common Industries:** [types]
   - **Required Skills:** [skills needed]

**3. [Job Title]**
   - **Why Suitable:** [reason]
   - **Common Industries:** [types]
   - **Required Skills:** [skills needed]

**💡 Skills to Develop:**
- [Skill 1]: [why important]
- [Skill 2]: [why important]

**📈 Career Tip:**
[Brief advice based on database patterns]

# RULES
✅ MUST use job_recommendation tool
✅ Only recommend jobs found in database
✅ Cite evidence (e.g., "8 of 15 professionals...")
✅ Be specific and data-driven
"""


# ===== SUPERVISOR PROMPT =====
SUPERVISOR_PROMPT = """
You are a Supervisor Agent for an AI Resume Screening system.

Analyze user queries and decide which specialized agent should handle the task.

# AVAILABLE AGENTS:
1. **searching_agent**: Finding/searching/matching candidates based on job requirements
2. **summarizer_agent**: Summarizing/analyzing resume information
3. **comparison_agent**: Comparing candidates based on IDs
4. **job_recommendation_agent**: Recommending jobs based on profile
5. **system_info**: Questions about capabilities/features (what can you do, help, etc.)
6. **direct_llm_response**: response general question, greeting, and query which doesn't relevant with the resume screening

# DECISION RULES:
- searching_agent: "find", "search", "recommend kandidat", "cari kandidat", "posisi", "looking for"
- summarizer_agent: "summarize", "ringkas", "analyze resume", "extract", "resume summary"
- comparison_agent: "compare", "bandingkan", "better", "worst", "versus", "vs"
- job_recommendation_agent: "cocok", "sesuai", "rekomendasi pekerjaan", "recommend job", "suitable job", "career", "karir"
- system_info: "apa yang bisa", "what can you do", "fitur apa", "kemampuan", "help", "bantuan", "how to use", "capabilities", "hal apa saja"
- direct_llm_response: "terima kasih", "selamat pagi", "selamat malam", "selamat sore", "selamat siang"

# INSTRUCTIONS:
Respond with ONLY ONE of: "searching_agent", "summarizer_agent", "comparison_agent", "job_recommendation_agent", "system_info", "direct_llm_response"
"""


# ===== MAIN CHATBOT FUNCTION =====
def run_chatbot(question: str, history: str):
    """
    Main function to run chatbot with supervisor logic
    """
    
    # Determine which agent to use
    supervisor_decision = llm.invoke(
        f"{SUPERVISOR_PROMPT}\n\nUser query: {question}"
    ).content.lower().strip()
    
    # Track metrics
    selected_agent = "None"
    tool_messages = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    supervisor_lower = supervisor_decision.lower().strip()
    
    # Handle system info question
    if "system_info" in supervisor_lower:
        selected_agent = "system_info"
        answer = """
**🤖 Tentang ResumeBot - AI Resume Screening Assistant**

Saya adalah asisten AI yang dirancang khusus untuk membantu HR profesional dalam proses screening dan pencarian kandidat.

---

**✅ Hal-hal yang Bisa Saya Lakukan:**

**1. 🔍 Mencari Kandidat Berdasarkan Persyaratan Pekerjaan**
   - Temukan kandidat dengan skills, pendidikan, atau pengalaman tertentu
   - Menggunakan AI semantic search untuk matching yang akurat
   - **Contoh:** *"Cari kandidat Python developer dengan 3 tahun pengalaman"*

**2. 📊 Membandingkan Multiple Kandidat**
   - Bandingkan 2 atau lebih kandidat secara detail
   - Analisis per kategori: pendidikan, skills, pengalaman, achievements
   - **Contoh:** *"Bandingkan kandidat ID 12345 dan 67890"*

**3. 📄 Merangkum Resume Kandidat**
   - Ringkas resume menjadi poin-poin penting yang terstruktur
   - Format: pendidikan, pengalaman, skills, pencapaian
   - **Contoh:** *"Ringkas resume kandidat ID 98765"*

**4. 💼 Rekomendasi Pekerjaan yang Cocok**
   - Berikan rekomendasi karir berdasarkan profil pendidikan dan skills
   - Analisis berbasis data real dari kandidat serupa di database
   - **Contoh:** *"Saya lulusan Teknik Kimia dengan skill Matlab, cocok kerja apa?"*

**5. 🎯 Analisis Mendalam Kandidat**
   - Jelaskan mengapa kandidat tertentu cocok untuk posisi
   - Identifikasi kekuatan unik dan nilai tambah kandidat
   - **Contoh:** *"Kenapa kandidat ID 12345 cocok untuk posisi Data Scientist?"*

---

**❌ Hal-hal yang Tidak Bisa Saya Lakukan:**

- ❌ Menulis atau membuat resume/CV untuk Anda
- ❌ Membuat deskripsi pekerjaan dari nol
- ❌ Memberikan tips wawancara atau coaching karir umum
- ❌ Mengakses data kandidat di luar database kami
- ❌ Membuat keputusan hiring final (saya hanya memberikan rekomendasi)
- ❌ Melakukan background check atau verifikasi kandidat

---

**📊 Sumber Data:**
Semua informasi kandidat berasal dari database Qdrant kami yang berisi resume embeddings. Saya menggunakan teknologi semantic search dengan AI untuk menemukan kandidat yang paling relevan.

---

**💡 Tips Penggunaan:**

✨ Untuk hasil terbaik, jelaskan requirement dengan spesifik:
   - ✅ *"Cari Software Engineer dengan Python, Django, 5 tahun pengalaman, pernah handle e-commerce"*
   - ⚠️ *"Cari programmer"* (terlalu umum)

✨ Untuk perbandingan kandidat, gunakan ID yang sudah Anda dapatkan dari pencarian sebelumnya

✨ Untuk rekomendasi job, sebutkan pendidikan, skills, dan pengalaman Anda

---

**🚀 Siap membantu Anda menemukan kandidat terbaik!**

Silakan mulai dengan pertanyaan spesifik Anda.
"""
        result = {
            "messages": [type('obj', (object,), {
                'content': answer,
                'response_metadata': {}
            })()]
        }
    
    # Route to specialized agents
    elif "job_recommendation_agent" in supervisor_lower:
        selected_agent = "job_recommendation_agent"
        agent = create_react_agent(model=llm, tools=[job_recommendation])
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"{JOB_RECOMMENDATION_PROMPT}\n\n**IMPORTANT: Answer ONLY the current question below. Ignore any previous questions in history.**\n\n**Current Question**: {question}\n\n**Previous Context (for reference only)**: {history}"
            }]
        })

    elif "comparison_agent" in supervisor_lower:
        selected_agent = "comparison_agent"
        agent = create_react_agent(model=llm, tools=[compare_candidates])
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"{COMPARISON_AGENT_PROMPT}\n\n**CRITICAL: You MUST use the candidate IDs from the CURRENT question only. DO NOT use IDs from chat history.**\n\n**Current Question**: {question}\n\n**Previous Context (for reference only)**: {history}"
            }]
        })
    
    elif "searching_agent" in supervisor_lower:
        selected_agent = "searching_agent"
        agent = create_react_agent(model=llm, tools=[get_ID])
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"{SEARCHING_AGENT_PROMPT}\n\n**IMPORTANT: Answer ONLY the current question below. Ignore any previous questions in history.**\n\n**Current Question**: {question}\n\n**Previous Context (for reference only)**: {history}"
            }]
        })
        
    elif "summarizer_agent" in supervisor_lower:
        selected_agent = "summarizer_agent"
        agent = create_react_agent(model=llm, tools=[summarize_resume_tool])
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"{SUMMARIZER_AGENT_PROMPT}\n\n**IMPORTANT: Answer ONLY the current question below. Ignore any previous questions in history.**\n\n**Current Question**: {question}\n\n**Previous Context (for reference only)**: {history}"
            }]
        })
    
    else:
        # Direct LLM response with system context
        selected_agent = "direct_llm_response"
        result = {
            "messages": [llm.invoke(
                f"{SYSTEM_CONTEXT}\n\n**User Question**: {question}\n\n**Conversation History**: {history}\n\nAnswer professionally based ONLY on your actual capabilities. Be truthful and helpful. If the question is outside your scope, politely explain what you CAN help with instead."
            )]
        }
    
    # Extract answer
    answer = result["messages"][-1].content
    
    # Extract tool messages
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage):
            tool_content = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
            tool_messages.append(f"Tool: {msg.name}\nResult: {tool_content}")
    
    # Calculate token usage
    for msg in result.get("messages", []):
        if hasattr(msg, 'response_metadata') and msg.response_metadata:
            if "usage_metadata" in msg.response_metadata:
                usage = msg.response_metadata["usage_metadata"]
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
            elif "token_usage" in msg.response_metadata:
                usage = msg.response_metadata["token_usage"]
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)
    
    # Calculate price
    price_usd = (total_input_tokens * 0.15 + total_output_tokens * 0.60) / 1_000_000
    price_idr = price_usd * 16_600
    price_str = f"Rp. {price_idr:,.2f}"
    
    return {
        "answer": answer,
        "selected_agent": selected_agent,
        "supervisor_decision": supervisor_decision,
        "tool_messages": "\n\n".join(tool_messages) if tool_messages else "No tools were used for this query.",
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "price": price_str
    }


# ===== STREAMLIT FRONTEND =====
st.set_page_config(
    page_title="AI Resume Screening Chatbot", 
    page_icon="🤖", 
    layout="wide"
)

st.title("💼 AI-Based Resume Screening Chatbot")

# Optional image with relative path
if os.path.exists("pic.png"):
    st.image("pic.png")
elif os.path.exists("D:\\chatbot\\pic.png"):
    st.image("D:\\chatbot\\pic.png")

st.markdown("""
<div style="text-align: center; font-size:24px;">
<b>WELCOME TO THE AI-BASED RESUME SCREENING CHATBOT! 🤖</b><br><br>
</div>
            
<div style="text-align: justify; font-size:18px;">
This intelligent chatbot is designed for HR professionals to streamline candidate screening and hiring decisions.<br><br>

<b>🎯 What You Can Do:</b><br>
✅ Find candidates matching specific job requirements<br>
✅ Compare multiple candidates side-by-side<br>
✅ Get resume summaries in structured format<br>
✅ Receive job recommendations based on profiles<br>
✅ Analyze why candidates match positions<br><br>

<b>📊 Powered by:</b> AI Semantic Search with Qdrant Vector Database<br><br>

💡 <b>Tip:</b> Try asking <i>"Apa yang bisa kamu lakukan?"</i> or <i>"What can you do?"</i> to see full capabilities!
</div>
""", unsafe_allow_html=True)

# Initialize session state (SAVE CONVERSATION)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "started" not in st.session_state:
    st.session_state.started = False


# ===== SIDEBAR - CHAT HISTORY =====
with st.sidebar:
    st.header("💬 Chat History")
    
    messages_history = st.session_state.get("messages", [])
    
    if messages_history:
        st.info(f"📊 Total Messages: {len(messages_history)}")
        
        # Group messages in pairs
        for i in range(0, len(messages_history), 2):
            if i < len(messages_history):
                user_msg = messages_history[i] if i < len(messages_history) else None
                assistant_msg = messages_history[i+1] if i+1 < len(messages_history) else None
                
                with st.expander(f"💭 Conversation {(i//2) + 1}", expanded=False):
                    if user_msg:
                        st.markdown("<b>👤 You:</b>", unsafe_allow_html=True)
                        st.info(user_msg["content"])
                    
                    if assistant_msg:
                        st.markdown("<b>🤖 ResumeBot:</b>", unsafe_allow_html=True)
                        st.success(assistant_msg["content"])
    else:
        st.warning("No conversation yet. Start chatting below! 👇")
    
# Display chat messages from history in main area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===== Starting Button Feature =====
if not st.session_state.started:
    if st.button("Click here to start"):
        st.session_state.started = True
        st.rerun()

# Accept user input
if st.session_state.started:
    if prompt := st.chat_input("Ask me about candidates or resume screening..."):
    
        # Only include LAST 4 messages to prevent confusion
        messages_history = st.session_state.get("messages", [])[-4:]  # Only last 4 messages
        history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or "No previous conversation"
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = run_chatbot(prompt, history)
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Display Tool Calls
        with st.expander("**Tool Calls:**"):
            st.code(response["tool_messages"])
        
        # Display Supervisor Decision
        with st.expander("**Supervisor Decision:**"):
            st.markdown(f"<b>🎯 Selected Agent:</b> <code>{response['selected_agent']}</code>", unsafe_allow_html=True)
            st.markdown(f"<b>🤔 Supervisor Reasoning:</b>", unsafe_allow_html=True)
            st.info(response['supervisor_decision'])
            
            # Show agent mapping for clarity
            st.markdown("**Agent Roles:**")
            st.markdown("- `searching_agent`: Find and match candidates based on job requirements")
            st.markdown("- `summarizer_agent`: Summarize and analyze resume content")
            st.markdown("- `comparison_agent`: Find ID given by the user and compare candidates based on their qualification")
            st.markdown("- `job_recommendation_agent`: Recommend suitable jobs based on given profile")
            st.markdown("- `direct_llm_response`: Handle general questions without tools")
            
        
        # Display Usage Details
        with st.expander("**Usage Details:**"):
            st.code(
                f"Input Tokens: {response['total_input_tokens']:,}\n"
                f"Output Tokens: {response['total_output_tokens']:,}\n"
                f"Estimated Cost: {response['price']}"
            )