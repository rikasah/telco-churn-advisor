"""Streamlit UI for Telco Churn Advisor: chat, analytics, and explainability."""
import os
import re

import altair as alt
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

# Telco customer IDs in this dataset always look like 7590-VHVEG (4 digits,
# dash, 5 letters). Used to detect when a chat message mentions a different
# customer, so the active customer switches automatically instead of
# requiring a manual edit of a separate box.
CUSTOMER_ID_PATTERN = re.compile(r"\b\d{4}-[A-Za-z]{5}\b")

# Mirrors backend/features.py CATEGORICAL + BOOLEAN -- the columns
# /aggregate accepts as group_by. Kept as a plain list here since the
# frontend runs in a separate container and doesn't import backend code.
GROUP_BY_OPTIONS = {
    "gender": "Gender",
    "contract": "Jenis Kontrak",
    "internet_service": "Jenis Layanan Internet",
    "payment_method": "Metode Pembayaran",
    "multiple_lines": "Multiple Lines",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "senior_citizen": "Senior Citizen",
    "partner": "Punya Partner",
    "dependents": "Punya Tanggungan",
    "phone_service": "Layanan Telepon",
    "paperless_billing": "Paperless Billing",
}

st.set_page_config(page_title="Telco Churn Advisor", layout="wide")
st.title("Telco Churn Advisor")

tab_chat, tab_analytics, tab_explain = st.tabs(["Chat", "Analytics", "Explain"])

# ============================================================
# Chat
# ============================================================
with tab_chat:
    # No visible/editable box: the active customer is tracked silently in
    # session_state and switches automatically whenever a message mentions
    # a customer ID, so there's nothing to manually reset between turns.
    if "active_customer_id" not in st.session_state:
        st.session_state.active_customer_id = "7590-VHVEG"

    st.caption(
        f"Sedang membahas pelanggan **{st.session_state.active_customer_id}**. Sebut "
        "Customer ID lain di pertanyaan (contoh: 'bagaimana dengan pelanggan "
        "9248-OJYKK') untuk otomatis ganti konteks."
    )
    st.caption(
        "Agent ini bisa lebih dari sekadar melaporkan skor satu pelanggan. Agent dapat "
        "menjelaskan alasannya, menjawab pertanyaan kebijakan, memberi statistik agregat "
        "seluruh basis pelanggan, atau sekadar menyapa. Coba salah satu contoh di bawah, "
        "atau ketik pertanyaan sendiri."
    )
    example_rows = [
        [
            "Kenapa saya berisiko pindah dari layanan ini?",
            "Apa saja penawaran retensi untuk pelanggan berisiko tinggi?",
            "Apa bedanya kontrak bulanan dan kontrak tahunan?",
        ],
        [
            "Ada berapa pelanggan yang berisiko tinggi churn?",
            "Sebutkan 5 pelanggan paling berisiko churn saat ini",
            "Berapa persen pelanggan pria dan wanita yang berpotensi churn?",
        ],
        [
            "Gimana churn rate berdasarkan jenis kontrak?",
            "Halo, apa kabar?",
        ],
    ]
    clicked_example = None
    for row in example_rows:
        cols = st.columns(len(row))
        for col, q in zip(cols, row):
            if col.button(q, use_container_width=True):
                clicked_example = q

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Render full history first, so the input box below always ends up
    # pinned at the very bottom of the conversation, not sandwiched
    # between old and new turns.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")
            if msg.get("tool_calls"):
                st.caption(f"Tools used: {', '.join(msg['tool_calls'])}")

    typed_prompt = st.chat_input(
        "Tanya soal customer ini, kebijakan retensi, atau apa saja..."
    )
    prompt = clicked_example or typed_prompt

    # Phase 1: a new question arrived. Just record it and rerun immediately
    # -- no network call yet -- so the user's message shows up and the input
    # box resets right away, instead of sitting blank while we wait on the
    # backend.
    if prompt and st.session_state.pending_prompt is None:
        id_match = CUSTOMER_ID_PATTERN.search(prompt)
        if id_match:
            st.session_state.active_customer_id = id_match.group(0).upper()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()

    # Phase 2: the actual (slower) backend call, running in its own script
    # pass so the chat input above has already been fully rendered before
    # this blocks.
    if st.session_state.pending_prompt is not None:
        with st.chat_message("assistant"):
            with st.spinner("Menjawab..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={
                            "customer_id": st.session_state.active_customer_id,
                            "message": st.session_state.pending_prompt,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": data.get("reply", ""),
                            "sources": data.get("sources", []),
                            "tool_calls": data.get("tool_calls", []),
                        }
                    )
                except requests.RequestException as e:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error calling backend: {e}"}
                    )
        st.session_state.pending_prompt = None
        st.rerun()

# ============================================================
# Analytics
# ============================================================
with tab_analytics:
    st.sidebar.header("Analytics")
    section = st.sidebar.radio(
        "Tampilkan",
        [
            "Ringkasan Sistem",
            "Trafik & Latency",
            "Distribusi Risiko & Tools",
            "Statistik per Kategori",
            "Tren Model (PR-AUC)",
            "Top-Risk Pelanggan",
        ],
        key="analytics_section",
    )
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.rerun()
    st.sidebar.caption("Semua angka diambil live dari Postgres/MLflow, bukan data statis.")

    st.caption("Ringkasan penggunaan sistem secara live, diambil dari log request nyata di Postgres.")

    if section == "Ringkasan Sistem":
        try:
            sys_status = requests.get(f"{BACKEND_URL}/system-status", timeout=15).json()
        except requests.RequestException as e:
            st.error(f"Tidak bisa mengambil status sistem: {e}")
            sys_status = None

        if sys_status:
            st.subheader("Kesehatan data & model")
            data_status = sys_status["data"]
            champion = sys_status["champion"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total pelanggan di database", f"{data_status['total_customers']:,}")
            last_ingested = data_status["last_ingested_at"]
            c2.metric("Data terakhir masuk", last_ingested[:19].replace("T", " ") if last_ingested else "-")
            job = data_status["last_ingestion_job"]
            job_status = job["status"] if job else "unknown"
            c3.metric("Status job ingestion terakhir", job_status)
            if champion:
                c4.metric("Model champion", f"v{champion['version']} (PR-AUC {champion['pr_auc']:.3f})")
            else:
                c4.metric("Model champion", "belum ada")

    elif section == "Trafik & Latency":
        try:
            stats = requests.get(f"{BACKEND_URL}/stats", timeout=15).json()
        except requests.RequestException as e:
            st.error(f"Tidak bisa mengambil statistik: {e}")
            stats = None

        if stats:
            st.subheader("Trafik")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total request", stats["total_requests"])
            col2.metric("/predict calls", stats["predict_count"])
            col3.metric("/chat calls", stats["chat_count"])

            latency = stats.get("latency", {})
            if latency:
                lat_cols = st.columns(len(latency))
                for col, (endpoint, l) in zip(lat_cols, latency.items()):
                    col.metric(f"Latency {endpoint} (avg / p95)", f"{l['avg_ms']:.0f}ms / {l['p95_ms']:.0f}ms")

            st.subheader("Request terakhir")
            if stats["recent"]:
                st.dataframe(pd.DataFrame(stats["recent"]), use_container_width=True)
            else:
                st.info("Belum ada request tercatat.")

    elif section == "Distribusi Risiko & Tools":
        try:
            stats = requests.get(f"{BACKEND_URL}/stats", timeout=15).json()
        except requests.RequestException as e:
            st.error(f"Tidak bisa mengambil statistik: {e}")
            stats = None

        if stats:
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

    elif section == "Statistik per Kategori":
        st.subheader("Churn rate berdasarkan kategori pelanggan")
        st.caption(
            "Dihitung on-demand dari model champion terhadap seluruh basis pelanggan. "
            "Pilih kategori di sidebar untuk melihat breakdown-nya."
        )
        group_by = st.sidebar.selectbox(
            "Kelompokkan berdasarkan",
            options=list(GROUP_BY_OPTIONS.keys()),
            format_func=lambda k: GROUP_BY_OPTIONS[k],
        )
        if st.button("Hitung statistik", key="aggregate_btn"):
            try:
                agg = requests.get(
                    f"{BACKEND_URL}/aggregate", params={"group_by": group_by}, timeout=60
                ).json()
            except requests.RequestException as e:
                st.error(f"Gagal menghitung: {e}")
                agg = None

            if agg and "error" in agg:
                st.error(agg["error"])
            elif agg:
                rows = [{"kategori": k, **v} for k, v in agg["hasil"].items()]
                df_agg = pd.DataFrame(rows)
                chart = (
                    alt.Chart(df_agg)
                    .mark_bar()
                    .encode(
                        x=alt.X("kategori:N", title=GROUP_BY_OPTIONS[group_by]),
                        y=alt.Y("persen_churn_aktual:Q", title="Persen churn aktual (%)"),
                        tooltip=["kategori", "jumlah_pelanggan", "persen_churn_aktual"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
                st.dataframe(df_agg, use_container_width=True)

    elif section == "Tren Model (PR-AUC)":
        st.subheader("Tren PR-AUC antar run training")
        try:
            trend = requests.get(f"{BACKEND_URL}/model-trend", timeout=15).json()
        except requests.RequestException as e:
            st.error(f"Tidak bisa mengambil riwayat training: {e}")
            trend = []
        if trend:
            df_trend = pd.DataFrame(trend)
            df_trend["run_order"] = range(1, len(df_trend) + 1)
            chart = (
                alt.Chart(df_trend)
                .mark_line(point=True)
                .encode(
                    x=alt.X("run_order:O", title="Urutan run"),
                    y=alt.Y("pr_auc:Q", title="PR-AUC", scale=alt.Scale(zero=False)),
                    tooltip=["run_name", "pr_auc"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Belum ada run training tercatat.")

    elif section == "Top-Risk Pelanggan":
        st.subheader("Pelanggan dengan risiko churn tertinggi saat ini")
        st.caption("Dihitung on-demand dari model champion terhadap seluruh basis pelanggan.")
        n_top = st.sidebar.slider("Jumlah pelanggan ditampilkan", 5, 30, 10)
        view = st.sidebar.radio("Tampilan", ["Tabel", "Grafik"], horizontal=True)
        if st.button("Hitung top-risk"):
            try:
                top_risk = requests.get(f"{BACKEND_URL}/top-risk", params={"n": n_top}, timeout=60).json()
                df_top = pd.DataFrame(top_risk)
                if view == "Tabel":
                    st.dataframe(df_top, use_container_width=True)
                else:
                    st.bar_chart(df_top, x="customer_id", y="churn_probability")
            except requests.RequestException as e:
                st.error(f"Gagal menghitung: {e}")

# ============================================================
# Explain
# ============================================================
with tab_explain:
    st.caption("Bongkar prediksi model untuk satu pelanggan menjadi faktor konkret (SHAP).")
    explain_customer_id = st.text_input("Customer ID untuk dijelaskan", value="7590-VHVEG", key="explain_cid")

    if st.button("Jelaskan prediksi"):
        try:
            pred_resp = requests.post(
                f"{BACKEND_URL}/predict", json={"customer_id": explain_customer_id}, timeout=30
            )
            pred_resp.raise_for_status()
            pred = pred_resp.json()
            st.metric("Churn probability", f"{pred['churn_probability']:.1%}", pred["risk_level"])

            exp_resp = requests.post(
                f"{BACKEND_URL}/explain", json={"customer_id": explain_customer_id}, timeout=60
            )
            exp_resp.raise_for_status()
            factors = exp_resp.json()["faktor_utama"]

            df_factors = pd.DataFrame(factors)
            df_factors["signed_impact"] = df_factors.apply(
                lambda r: r["besar_pengaruh"] if r["arah"] == "meningkatkan risiko" else -r["besar_pengaruh"],
                axis=1,
            )
            df_factors["label"] = df_factors["faktor"] + " = " + df_factors["nilai_pelanggan"]

            increasing = [f for f in factors if f["arah"] == "meningkatkan risiko"]
            decreasing = [f for f in factors if f["arah"] == "menurunkan risiko"]

            def _factor_list(flist):
                return ", ".join(f"{f['faktor']} ({f['nilai_pelanggan']})" for f in flist)

            summary = (
                f"Pelanggan **{explain_customer_id}** memiliki probabilitas churn sebesar "
                f"**{pred['churn_probability']:.1%}**, masuk kategori risiko **{pred['risk_level']}**."
            )
            if increasing:
                summary += f" Faktor yang paling meningkatkan risiko: {_factor_list(increasing)}."
            if decreasing:
                summary += f" Faktor yang paling menurunkan risiko: {_factor_list(decreasing)}."
            st.markdown(summary)

            chart = (
                alt.Chart(df_factors)
                .mark_bar()
                .encode(
                    x=alt.X("signed_impact:Q", title="Pengaruh terhadap risiko churn"),
                    y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
                    color=alt.condition(
                        alt.datum.signed_impact > 0,
                        alt.value("#C94A4A"),
                        alt.value("#3E9B5C"),
                    ),
                    tooltip=["faktor", "nilai_pelanggan", "arah", "besar_pengaruh"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Merah = meningkatkan risiko churn. Hijau = menurunkan risiko churn.")
        except requests.RequestException as e:
            st.error(f"Gagal menjelaskan prediksi: {e}")
