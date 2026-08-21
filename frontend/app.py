"""Streamlit chat UI for Telco Churn Advisor. Skeleton only."""
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Telco Churn Advisor")
st.title("Telco Churn Advisor")

customer_id = st.text_input("Customer ID", value="C001")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Tanya sesuatu tentang customer ini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={"customer_id": customer_id, "message": prompt},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("reply", "")
            st.write(reply)
            if data.get("sources"):
                st.caption(f"Sources: {', '.join(data['sources'])}")
        except requests.RequestException as e:
            reply = f"Error calling backend: {e}"
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
