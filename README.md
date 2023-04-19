## Overview

The Agent creates a Todo list to complete tasks that are required by the user and even searches Google for you.
This is a BabyAGI + SerpAPI tool that uses Langchain's framework. The site is hosted by Streamlit.

---

## To run

Make sure to fill in the API keys for OpenAI and SerpAPI.

```
python -m venv venv
source venv/bin/activate
pip install langchain
pip install openai
pip install google-search-results
pip install faiss-cpu > /dev/null
pip install tiktoken
pip install streamlit
streamlit run agent.py

```
If streamlit is not working, checkout their [installation page](https://docs.streamlit.io/library/get-started/installation)

---

### Develop Log

23-04-18

1. BabyAGI + serpapi Agent

2. Frontend UI on streamlit
   - User input
   - Download results as one .txt file

---

### TODO

- [ ] Need to revise the prompts for more precise tasks

---
