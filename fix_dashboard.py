#!/usr/bin/env python3
"""Fix dashboard.py DESIGN.md compliance issues."""

import sys

# Read the file
with open('app/ui/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the metrics HTML - replace the broken f-string HTML with separate card variables
# Find and replace the metrics section

# First, let's find the exact location and replace the broken card definitions
# We'll replace the entire metrics row section

# The problem: f-strings with inner double quotes break the HTML
# Solution: Use separate card variable definitions

# Replace the c1-c5 block and add card variables before it
old_section = """    with c1:
            st.markdown(f\"\"\"
            <div style="
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 9px;
                padding: 16px 12px;
                margin-bottom: 8px;
                font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif';
                letter-spacing: -0.374px;
            ">
                <div style="font-size: 14px; color: #8f95a3; margin-bottom: 4px;">SSIM Structural Match</div>
                <div style="font-size: 24px; font-weight: 600; color: #1d1d1f;">{results['ssim_score']:.3f}</div>
            </div>\"\"\", unsafe_allow_html=True)

        with c2:
            st.markdown(f\"\"\"
            <div style="
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 9px;
                padding: 16px 12px;
                margin-bottom: 8px;
                font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif';
                letter-spacing: -0.374px;
            ">
                <div style="font-size: 14px; color: #8f95a3; margin-bottom: 4px;">CV Anomaly Regions</div>
                <div style="font-size: 24px; font-weight: 600; color: #1d1d1f;">{results['defect_count']}</div>
            </div>\"\"\", unsafe_allow_html=True)

        with c3:
            st.markdown(f\"\"\"
            <div style="
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 9px;
                padding: 16px 12px;
                margin-bottom: 8px;
                font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif';
                letter-spacing: -0.374px;
            ">
                <div style="font-size: 14px; color: #8f95a3; margin-bottom: 4px;">AI Defect Probability</div>
                <div style="font-size: 24px; font-weight: 600; color: #1d1d1f;">{results.get('defect_prob', 0.0)*100:.1f}%</div>
            </div>\"\"\", unsafe_allow_html=True)

        with c4:
            st.markdown(f\"\"\"
            <div style="
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 9px;
                padding: 16px 12px;
                margin-bottom: 8px;
                font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif';
                letter-spacing: -0.374px;
            ">
                <div style="font-size: 14px; color: #8f95a3; margin-bottom: 4px;">Pin Count & Pitch</div>
                <div style="font-size: 24px; font-weight: 600; color: #1d1d1f;">{pin_res['total_pins']}/16 Pins</div>
                <div style="font-size: 13px; color: #8f95a3;">Pitch: {pin_res['pitch_mean']:.1f}px</div>
            </div>\"\"\", unsafe_allow_html=True)

        with c5:
            alignment_text = "Homography Aligned" if results["is_aligned"] else "Standard Match"
            st.markdown(f\"\"\"
            <div style="
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 9px;
                padding: 16px 12px;
                margin-bottom: 8px;
                font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif';
                letter-spacing: -0.374px;
            ">
                <div style="font-size: 14px; color: #8f95a3; margin-bottom: 4px;">Alignment</div>
                <div style="font-size: 24px; font-weight: 600; color: #1d1d1f;">{alignment_text}</div>
            </div>\"\"\", unsafe_allow_html=True)"""

new_section = """    global ssim_card, defect_card, prob_card, pin_card, alignment_card
    
    ssim_card = \"\"\"<div style=\"background: #ffffff; border: 1px solid #e5e7eb; border-radius: 9px; padding: 16px 12px; margin-bottom: 8px; font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif'; letter-spacing: -0.374px;\">
                <div style=\"font-size: 14px; color: #8f95a3; margin-bottom: 4px;'>SSIM Structural Match</div>
                <div style=\"font-size: 24px; font-weight: 600; color: #1d1d1f;\">{results['ssim_score']:.3f}</div>
            </div>\"\"\"
    
    defect_card = \"\"\"<div style=\"background: #ffffff; border: 1px solid #e5e7eb; border-radius: 9px; padding: 16px 12px; margin-bottom: 8px; font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif'; letter-spacing: -0.374px;\">
                <div style=\"font-size: 14px; color: #8f95a3; margin-bottom: 4px;'>CV Anomaly Regions</div>
                <div style=\"font-size: 24px; font-weight: 600; color: #1d1d1f;\">{results['defect_count']}</div>
            </div>\"\"\"
    
    prob_card = \"\"\"<div style=\"background: #ffffff; border: 1px solid #e5e7eb; border-radius: 9px; padding: 16px 12px; margin-bottom: 8px; font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif'; letter-spacing: -0.374px;\">
                <div style=\"font-size: 14px; color: #8f95a3; margin-bottom: 4px;'>AI Defect Probability</div>
                <div style=\"font-size: 24px; font-weight: 600; color: #1d1d1f;\">{results.get('defect_prob', 0.0)*100:.1f}%</div>
            </div>\"\"\"
    
    pin_card = \"\"\"<div style=\"background: #ffffff; border: 1px solid #e5e7eb; border-radius: 9px; padding: 16px 12px; margin-bottom: 8px; font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif'; letter-spacing: -0.374px;\">
                <div style=\"font-size: 14px; color: #8f95a3; margin-bottom: 4px;'>Pin Count & Pitch</div>
                <div style=\"font-size: 24px; font-weight: 600; color: #1d1d1f;\">{pin_res['total_pins']}/16 Pins</div>
                <div style=\"font-size: 13px; color: #8f95a3;\">Pitch: {pin_res['pitch_mean']:.1f}px</div>
            </div>\"\"\"
    
    alignment_card = \"\"\"<div style=\"background: #ffffff; border: 1px solid #e5e7eb; border-radius: 9px; padding: 16px 12px; margin-bottom: 8px; font-family: 'SF Pro Text, system-ui, -apple-system, sans-serif'; letter-spacing: -0.374px;\">
                <div style=\"font-size: 14px; color: #8f95a3; margin-bottom: 4px;'>Alignment</div>
                <div style=\"font-size: 24px; font-weight: 600; color: #1d1d1f;\">{alignment_text}</div>
            </div>\"\"\"

    with c1:
        st.markdown(ssim_card, unsafe_allow_html=True)
    with c2:
        st.markdown(defect_card, unsafe_allow_html=True)
    with c3:
        st.markdown(prob_card, unsafe_allow_html=True)
    with c4:
        st.markdown(pin_card, unsafe_allow_html=True)
    with c5:
        st.markdown(alignment_card, unsafe_allow_html=True)"""

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('app/ui/dashboard.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Metrics section fixed successfully")
else:
    print("Old section not found - trying alternative approach")
    # Try to just fix the specific broken lines
    print("Content length:", len(content))
    # Show what's around line 299
    lines = content.split('\n')
    for i in range(295, min(310, len(lines))):
        print(f'{i+1}: {lines[i][:100]}')