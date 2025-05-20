import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True

import streamlit as st
from utils import Utils, LOG_HANDLERS, IDENTIFIERS, DEFAULT_HEADER, LOG_FILE_PATH



def render_system_logs():
    cols=st.columns([1,2,2,1])
    placeholder_content=st.container(border=True, height=950)
    cols[3].write("")
    cols[3].write("")
    beautify=cols[3].toggle("Beautify Logs", help="Make the logs more colorful but it may impact performance if many entries.")
    
    cols[0].header("System Logs")
    cols[1].write("")
    cols[1].write("")
    if cols[1].button("View Recent Logs", type="tertiary", use_container_width=True):
        # Load logs from file at startup
        system_logs = Utils.load_system_logs_from_file()
        with placeholder_content:
            # Clear previous logs display if needed, then display updated logs
            for i in range(len(system_logs) - 1, -1, -1):
                log = system_logs[i]
                # Remove prefix for cleaner display
                for identifier in IDENTIFIERS:
                    if log.startswith(identifier):
                        log = log.replace(identifier, "")
                        break
                # Use handler and icon for each log line
                h, ic = next(
                    ((func, icon) for identifier, (func, icon) in LOG_HANDLERS.items() if system_logs[i].startswith(identifier)),
                    DEFAULT_HEADER
                )
                if beautify:
                    h(f"-> [{i+1}] - {log}", icon=ic)
                else:
                    st.write(f"{ic} -> [{i+1}] - {log}")
                    divs=st.columns([1,3,1])
                    divs[1].divider()

    cols[2].write("")
    cols[2].write("")
    if cols[2].button("Clear Logs", type="tertiary", use_container_width=True):
        st.session_state.system_logs=[]
        # Clear the content of the log file if it exists
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                pass  # Just opening in 'w' mode truncates the file

