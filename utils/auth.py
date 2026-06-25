"""Simple password gate for Streamlit."""

from __future__ import annotations

import os

import streamlit as st

from config.settings import STREAMLIT_PASSWORD_ENV


def require_auth() -> bool:
    """
    Returns True if authenticated. Set STREAMLIT_PASSWORD in .env to enable.
    When unset, auth is disabled (dev mode).
    """
    expected = os.getenv(STREAMLIT_PASSWORD_ENV)
    if not expected:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title("RAG Retrieval Lab — Login")
    password = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False
