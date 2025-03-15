import streamlit as st
st.set_page_config(page_title="Home", page_icon="🏠")
# -------- Header Section --------
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Machine Learning & Neural Network Web Application</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center; color: #888;'>Intelligent System Course Project</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------- Description Section --------
st.markdown("""
<div style='font-size:18px; text-align: justify;'>
This web application is developed as part of the Intelligent System course 
to demonstrate the process of building <span style='color: #4CAF50; font-weight: bold;'>Machine Learning</span> and <span style='color: #4CAF50; font-weight: bold;'>Neural Network</span> 
models — starting from <b>data preparation</b>,  
<b>algorithm theory</b>, and <b>model development steps</b> to deploying and testing the models interactively.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("<h4 style='color:#4CAF50; text-align: center'>Website Sections</h4>", unsafe_allow_html=True)
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/page1.py", label="ML Model Development", icon="📘")
    st.page_link("pages/page2.py", label="ML Model Demonstration", icon="🔍")

with col2:
    st.page_link("pages/page3.py", label="Neural Network Model Development", icon="📗")
    st.page_link("pages/page4.py", label="Neural Network Model Demonstration", icon="🔎")

st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("<p style='text-align: center; font-size:16px;'>Developed by: Fadia Hajeeyusoh | Student ID: 6604062630391</p>", unsafe_allow_html=True)


