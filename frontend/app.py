# frontend/app.py
import streamlit as st
import requests

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

st.title("SmartSearch Demo")
query = st.text_input("Enter your search query:")

if st.button("Search"):
    resp = requests.get(f"{API_URL}/search", params={"q": query, "k": 5})
    if resp.status_code == 200:
        for r in resp.json()["results"]:
            st.markdown(f"**{r['title']}**  \n{r['snippet']}...  \n_Score: {r['score']:.2f}_")
    else:
        st.error("Search failed.")
