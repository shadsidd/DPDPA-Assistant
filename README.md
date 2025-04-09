# DPDPA UI Application

A Streamlit-based interactive Q&A assistant for understanding India's Digital Personal Data Protection Act (DPDPA). This application provides both simple and detailed explanations of DPDPA-related queries using AI-powered agents and knowledge base integration.

## Features

- 🤖 AI-powered DPDPA expert assistant
- 🌐 Real-time internet search capabilities for latest updates
- 💡 Dual response modes:
  - Simple Mode: Clear, concise explanations
  - Detailed Mode: Comprehensive, in-depth analysis
- 📚 Integrated knowledge base using ChromaDB
- 🔍 Source attribution and reference tracking
- 💬 Interactive chat interface

## Prerequisites

- Python 3.7+
- ChromaDB
- Streamlit
- Agno Framework and its dependencies
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd dpdpa-ui-application
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure ChromaDB is properly configured with the collection:
```python
COLLECTION_NAME = "dpdpa_knowledge_lc_final_v5"
CHROMA_PERSIST_DIR = "./dpdpa_chroma_lc_final_v5"
```

## Usage

1. Start the application:
```bash
streamlit run dpdpa_streamlit_lite.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Use the interface to:
   - Ask questions about DPDPA
   - Toggle between Simple and Detailed response modes
   - View source references
   - Access internet search for latest updates
   - Clear chat history as needed

## Project Structure

```
DPDPA UI Application/
├── dpdpa_streamlit_lite.py  # Main application file
├── dpdpa_chroma_lc_final_v5/  # ChromaDB persistence directory
└── requirements.txt  # Project dependencies
```

## Features in Detail

### AI Agents

1. DPDPA Expert Agent
   - Provides clear explanations of DPDPA concepts
   - Uses structured response format with headers
   - Includes source citations

2. Internet Search Agent
   - Fetches latest DPDPA updates
   - Provides web-sourced information
   - Includes source URLs and dates

### Response Modes

1. Simple Mode
   - Clear, 1-sentence summaries
   - Structured explanations
   - Practical implications
   - "In Simple Terms" summary

2. Detailed Mode
   - Comprehensive analysis
   - Multiple sub-sections
   - Technical details
   - Regulatory nuances

## Security and Disclaimers

- The application is for informational purposes only
- Not a substitute for professional legal advice
- Always verify critical legal information
- Consult qualified legal experts for specific DPDPA compliance guidance

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Add your license information here]

## Support

For support, please [add contact information or support channels] 