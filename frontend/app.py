"""Streamlit UI for Telco Churn Advisor: chat + a small analytics view."""
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Telco Churn Advisor", layout="wide")
st.title("Telco Churn Advisor")

tab_chat, tab_analytics = st.tabs(["Chat", "Analytics"])

with tab_chat:
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
                if data.get("tool_calls"):
                    st.caption(f"Tools used: {', '.join(data['tool_calls'])}")
            except requests.RequestException as e:
                reply = f"Error calling backend: {e}"
                st.error(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

with tab_analytics:
    st.caption("Ringkasan penggunaan sistem secara live -- diambil dari log request nyata di Postgres.")

    if st.button("Refresh"):
        st.rerun()

    try:
        stats = requests.get(f"{BACKEND_URL}/stats", timeout=15).json()
    except requests.RequestException as e:
        st.error(f"Tidak bisa mengambil statistik: {e}")
        stats = None

    if stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total request", stats["total_requests"])
        col2.metric("/predict calls", stats["predict_count"])
        col3.metric("/chat calls", stats["chat_count"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribusi risk level")
            risk = stats["risk_level_distribution"]
            if risk:
                df_risk = pd.DataFrame({"risk_level": list(risk.keys()), "jumlah": list(risk.values())})
                st.bar_chart(df_risk, x="risk_level", y="jumlah")
            else:
                st.info("Belum ada data /predict.")

        with col_b:
            st.subheader("Tools yang paling sering dipanggil agent")
            tools = stats["top_tools"]
            if tools:
                df_tools = pd.DataFrame({"tool": list(tools.keys()), "jumlah": list(tools.values())})
                st.bar_chart(df_tools, x="tool", y="jumlah")
            else:
                st.info("Belum ada data /chat dengan tool call.")

        st.subheader("Request terakhir")
        if stats["recent"]:
            st.dataframe(pd.DataFrame(stats["recent"]), use_container_width=True)
        else:
            st.info("Belum ada request tercatat.")
