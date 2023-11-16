import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
from utils import *

########## Title for the Web App ##########
st.title("Text Classification for Service Feedback")

########## Create Input field ##########
feedback = st.text_input('Type your text here', 'The website was user friendly and the agent provided good solutions')

if st.button('Click for predictions!'):
    with st.spinner('Generating predictions...'):
        
        topics_prob, sentiment_prob = get_single_prediction(feedback)
        
        bar = px.bar(topics_prob, x='probability', y='topic')
                
        pie = px.pie(sentiment_prob, 
               values='probability', 
               names='sentiment', 
               color_discrete_map={'positive':'rgb(0, 204, 0)', 
                                 'negative':'rgb(215, 11, 11)'
                                  },
               color='sentiment'
              )
        
    st.plotly_chart(bar, use_container_width=True)    
    st.plotly_chart(pie, use_container_width=True)

st.write("\n")    
st.subheader('Or... Upload a csv file if you have a file instead.')
st.write("\n")

st.download_button(
     label="Download sample file here",
     data=sample_file,
     file_name='sample_data.csv',
     mime='text/csv',
 )

uploaded_file = st.file_uploader("Please upload a csv file with only 1 column of texts.")

if uploaded_file is not None:
    
    with st.spinner('Generating predictions...'):
        results = get_multiple_predictions(uploaded_file)
    
    st.download_button(
     label="Download results as CSV",
     data=results,
     file_name='results.csv',
     mime='text/csv',
     )
    
    