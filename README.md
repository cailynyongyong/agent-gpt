## Overview

This is a BabyAGI + SerpAPI tool embedded site that is hosted by Streamlit.
The Agent creates a Todo list to complete tasks that are required by the user.

---

## To run

```
python -m venv venv
source venv/bin/activate
pip install langchain
pip install google-search-results
pip install faiss-cpu > /dev/null
pip install tiktoken
pip install streamlit
streamlit run BabyAGI.py

```

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
