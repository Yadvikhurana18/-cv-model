#!/usr/bin/env python3
import sys

# Read the file
with open('app/ui/dashboard.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace the cache_resource decorators with regular function definitions
old_pattern = """@st.cache_resource
def get_pipeline():
    return InspectionPipeline()

@st.cache_resource
def get_logger():
    return InspectionLogger()

@st.cache_resource
def get_pin_analyzer():
    return PinAnalyzer()

@st.cache_resource
def get_reporter():
    return ReportGenerator()

pipeline = get_pipeline()
logger = get_logger()
pin_analyzer = get_pin_analyzer()
reporter = ReportGenerator()"""

new_pattern = """def get_pipeline():
    return InspectionPipeline()

def get_logger():
    return InspectionLogger()

def get_pin_analyzer():
    return PinAnalyzer()

def get_reporter():
    return ReportGenerator()

# Initialize resources (will be called from main app flow)
pipeline = get_pipeline()
logger = get_logger()
pin_analyzer = get_pin_analyzer()
reporter = ReportGenerator()"""

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open('app/ui/dashboard.py', 'w', encoding='utf-8', errors='replace') as f:
        f.write(content)
    print("Fixed cache_resource decorators successfully")
    sys.exit(0)
else:
    print("Pattern not found in file")
    # Debug: show what's around line 1-50
    lines = content.split('\n')[:50]
    for i, line in enumerate(lines, 1):
        print(f"{i}: {line[:80]}")
sys.exit(1)