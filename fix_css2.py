#!/usr/bin/env python3
with open('app/ui/dashboard.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

old_section = """# Custom Design System - APPLE DESIGN.MD COMPLIANT
# Custom Design System - APPLE DESIGN.MD COMPLIANT
st.markdown(
    """
    <style>
    :root {
        --primary: "#0066cc";
        --primary-focus: "#0071e3";
        --primary-on-dark: "#2997ff";
        --ink: "#1d1d1f";
        --body: "#1d1d1f";
        --body-on-dark: "#ffffff";
        --body-muted: "#cccccc";
        --ink-muted-80: "#333333";
        --ink-muted-48: "#7a7a7a";
        --divider-soft: "#f0f0f0";
        --hairline: "#e0e0e0";
        --canvas: "#ffffff";
        --canvas-parchment: "#f5f5f7";
        --surface-pearl: "#fafafc";
        --surface-tile-1: "#272729";
        --surface-tile-2: "#2a2a2c";
        --surface-tile-3: "#252527";
        --surface-black: "#000000";
        --surface-chip-translucent: "#d2d2d7";
        --on-primary: "#ffffff";
        --on-dark: "#ffffff";
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    .main, .stApp, html, body {
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif", Inter, sans-serif !important;
        color: var(--ink) !important;
    }
    .main { background-color: transparent; }

    .stButton > button[kind="primary"] {
        background: var(--primary) !important;
        color: var(--on-primary) !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 11px 22px !important;
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif" !important;
        font-size: 17px !important;
        font-weight: 400 !important;
        letter-spacing: -0.374px !important;
        margin: 4px 0 !important;
        transition: transform 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: scale(0.95) !important;
    }
    .stButton > button[kind="primary"]:focus {
        outline: 2px solid var(--primary-focus) !important;
        outline-offset: 2px !important;
    }

    .stButton > button[kind="secondary"] {
        background: var(--canvas) !important;
        color: var(--ink) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 9999px !important;
        padding: 11px 22px !important;
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif" !important;
        font-size: 17px !important;
        font-weight: 400 !important;
        letter-spacing: -0.374px !important;
        margin: 4px 0 !important;
        transition: transform 0.15s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        transform: scale(0.95) !important;
    }

    .stButton [data-primary="true"] {
        background: var(--ink) !important;
        color: var(--body-on-dark) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 15px !important;
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif" !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        letter-spacing: -0.224px !important;
        margin: 4px 0 !important;
    }

    .metric-card {
        background: var(--canvas) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 9px !important;
        padding: 16px 12px !important;
        margin-bottom: 8px !important;
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif" !important;
        letter-spacing: -0.374px !important;
        color: var(--ink) !important;
    }
    .metric-card .label {
        font-size: 13px !important;
        color: var(--body-muted) !important;
        margin-bottom: 6px !important;
    }
    .metric-card .value {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: var(--ink) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 12px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        border-radius: 0px !important;
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif" !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        letter-spacing: -0.224px !important;
        background: transparent !important;
        border: none !important;
        color: var(--ink-muted-80) !important;
        transition: color 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    .verdict-pass {
        background: #ffffff !important;
        color: var(--primary) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 9999px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 26px !important;
        text-align: center !important;
        letter-spacing: -0.28px !important;
        margin: 16px 0 !important;
    }
    .verdict-fail {
        background: #fecaca !important;
        color: var(--primary) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 9999px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 26px !important;
        text-align: center !important;
        letter-spacing: -0.28px !important;
        margin: 16px 0 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 12px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        border-radius: 0px !important;
        font-family: "SF Pro Text, system-ui, -apple-system, sans-serif" !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        letter-spacing: -0.224px !important;
        background: transparent !important;
        border: none !important;
        color: var(--ink-muted-80) !important;
        transition: color 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }
    """
)
"""

new_section = """# Custom Design System - APPLE DESIGN.MD COMPLIANT
# Note: CSS styling should be initialized in the Streamlit app context.
# The init_design_system() function provides the DESIGN.MD compliant styles.
# Uncomment the following line in your app.py to initialize:
# from app.ui.dashboard import init_design_system
# init_design_system()"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('app/ui/dashboard.py', 'w', encoding='utf-8', errors='replace') as f:
        f.write(content)
    print("Successfully removed module-level st.markdown and added comment note")
else:
    print("Section not found - showing current state")
    # Debug: print first 300 chars
    print(content[:300])
PYEOF