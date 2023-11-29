import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
from io import StringIO
import requests  

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Streamlit page configuration
st.set_page_config(page_title="Welcome to CrowdTangle Analyser", layout="wide")

# CSS for centering the title and customizing its style
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 50px; /* Adjust the font size as needed */
        color: skyblue; /* Change the font color to sky blue */
        margin-bottom: 20px; /* Add some spacing below the title */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the app
st.markdown("<h1 class='centered-title'>CrowdTangle Analysis</h1>", unsafe_allow_html=True)

# File uploader for the CSV file
uploaded_csv = st.file_uploader("Upload a CSV file", type="csv")

# Function to clean text
def clean_text(text, stop_words=None):
    if not isinstance(text, str):
        return ''
    text = ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])
    if stop_words:
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to extract hashtags
def extract_hashtags(text, stop_words=None):
    if not isinstance(text, str):
        return []
    hashtags = re.findall(r'#\w+', text)
    if stop_words:
        filtered_hashtags = [hashtag for hashtag in hashtags if hashtag not in stop_words]
    else:
        filtered_hashtags = hashtags
    return filtered_hashtags

# Word Cloud Generator Section

st.header("Word Cloud Generator")
uploaded_txt_wc = st.file_uploader("Upload an optional text file for stop words (Word Cloud)", type="txt", accept_multiple_files=False)

if st.button('Generate Word Cloud'):
    if uploaded_csv is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_csv)

        # # Define the URL of the stop words file on GitHub
        # stopwords_url = 'https://raw.githubusercontent.com/Adam0112/CT/main/stopwords.txt'

        # Fetch the contents of the stop words file
        response = requests.get(stopwords_url)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Extract the stop words as a set
            custom_stop_words = set(response.text.splitlines())
        else:
            # If the request fails, use an empty set
            custom_stop_words = set()

        # Combine NLTK's English stopwords with custom stop words
        stop_words = set(stopwords.words('english')).union(custom_stop_words)

        # Apply cleaning function
        df['cleaned_text'] = df['Message'].apply(lambda x: clean_text(x, stop_words))

        # Generate word cloud
        all_text = ' '.join(df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        # Display word cloud
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to generate the word cloud.")

st.markdown("<br><br>", unsafe_allow_html=True)

# Hashtag Extraction Section
st.header("Extract Hashtags")
uploaded_txt_hashtags = st.file_uploader("Upload an optional text file for stop words (Hashtags)", type="txt", accept_multiple_files=False)
number_of_hashtags = st.number_input("Number of Hashtags to Display", min_value=1, max_value=100, value=20)

if st.button('Extract Hashtags'):
    if uploaded_csv is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_csv)

        # Prepare stop words for Hashtags
        hashtag_stop_words = set()
        if uploaded_txt_hashtags is not None:
            hashtag_stop_words = set(uploaded_txt_hashtags.getvalue().decode().splitlines())

        # Extract and visualize hashtags
        hashtags_list = df['text'].apply(lambda x: extract_hashtags(x, hashtag_stop_words)).explode()
        top_hashtags = hashtags_list.value_counts().nlargest(number_of_hashtags)

        if not top_hashtags.empty:
            # Create a horizontal bar chart for hashtags
            plt.figure(figsize=(10, 6))
            top_hashtags.plot(kind='barh', color='skyblue')
            plt.gca().invert_yaxis()  # Reverse the order for better visualization
            plt.title(f'Top {number_of_hashtags} Most Frequent Hashtags')
            plt.xlabel('Frequency')
            plt.ylabel('Hashtag')
            st.pyplot(plt)
        else:
            st.write("No hashtags found in the data.")
    else:
        st.write("Please upload a CSV file to extract hashtags.")

st.markdown("<br><br>", unsafe_allow_html=True)

# Time Series Analysis Section
st.header("Time Series Analysis of Engagement")

# Allow user to specify the interval
interval = st.slider("Select the Interval", min_value=1, max_value=30, value=7)

if st.button('Analyse Time Series Engagement'):
    if uploaded_csv is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_csv).copy()

        # Convert the 'Post Created Date' column to datetime
        df['Post Created Date'] = pd.to_datetime(df['Post Created Date'], errors='coerce')

        # Handling missing values (if any)
        df.fillna(0, inplace=True)

        # Aggregating the data by the specified interval
        daily_data = df.groupby(df['Post Created Date'].dt.date).sum()
        daily_data.fillna(0, inplace=True)

        # Plotting time series for all engagement metrics in one graph
        plt.figure(figsize=(15, 8))

        metrics = ['Likes', 'Comments', 'Shares', 'Love', 'Sad', 'Angry']
        for metric in metrics:
            sns.lineplot(data=daily_data, x=daily_data.index, y=metric, label=metric)

        plt.title('Facebook Post Engagement Over Time')
        plt.ylabel('Count')
        plt.xlabel('Date')

        # Set x-axis to show more dates based on the selected interval
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.legend(title='Engagement Types')
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("Please upload a CSV file to analyze time series engagement.")

# Hourly Distribution Analysis
st.markdown("<br><br>", unsafe_allow_html=True)

st.header("Post Distribution by Hour")

if st.button('Analyse Hourly Post Distribution'):
    if uploaded_csv is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_csv).copy()
        # Parse the "Post Created Time" and extract hour
        df['Hour'] = pd.to_datetime(df['Post Created Time'], format='%H:%M:%S').dt.hour

        # Count posts in each hour
        hourly_counts = df['Hour'].value_counts().sort_index()

        # Plotting
        plt.figure(figsize=(12, 6))
        ax = hourly_counts.plot(kind='bar', zorder=3)
        plt.title('Facebook Posts Distribution by Hour of the Day')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Posts')
        plt.xticks(range(24), [f'{hour:02d}:00' for hour in range(24)], rotation=45)
        plt.grid(True, zorder=0)
        st.pyplot(plt)
    else:
        st.write("Please upload a CSV file.")


# st.header("Posting Time")

# if st.button('Posting Time'):
#     if uploaded_csv is not None:
#         # Read CSV file
#         df = pd.read_csv(uploaded_csv).copy()
#         # Parse the "Post Created Time" and extract hour
#         df['Hour'] = pd.to_datetime(df['Post Created Time'], format='%H:%M:%S').dt.hour

#         # Group by page name and hour, then count posts
#         grouped_counts = df.groupby(['User Name', 'Hour']).size().unstack(fill_value=0)

#         # Plotting
#         plt.figure(figsize=(12, 6))
#         grouped_counts.T.plot(kind='bar', stacked=True, zorder=3, ax=plt.gca())
#         plt.title('Facebook Posts Distribution by Hour of the Day Across Pages')
#         plt.xlabel('Hour of the Day')
#         plt.ylabel('Number of Posts')
#         plt.xticks(range(24), [f'{hour:02d}:00' for hour in range(24)], rotation=45)
#         plt.legend(title='Page Name')
#         plt.grid(True, zorder=0)
#         st.pyplot(plt)
#     else:
#         st.write("Please upload a CSV file.")

# if st.button('Analyse Hourly Post Distribution'):
#     if uploaded_csv is not None:
#         # Read CSV file
#         df = pd.read_csv(uploaded_csv).copy()
#         # Parse the "Post Created Time" and extract hour
#         df['Hour'] = pd.to_datetime(df['Post Created Time'], format='%H:%M:%S').dt.hour

#         # Filter hours from 09 to 17
#         target_hours = df[df['Hour'].between(9, 17)]

#         # Group by page name and hour, then count posts
#         total_counts = df.groupby('User Name').size()
#         target_counts = target_hours.groupby('Page Name').size()

#         # Calculate the percentage of posts in target hours
#         percentage = (target_counts / total_counts) * 100

#         # Filter pages with at least 90% of posts in target hours
#         target_pages = percentage[percentage >= 90].index

#         # Filter the original dataframe for the target pages
#         filtered_df = df[df['User Name'].isin(target_pages)]

#         # Group by page name and hour, then count posts for filtered pages
#         grouped_counts = filtered_df.groupby(['User Name', 'Hour']).size().unstack(fill_value=0)

#         # Plotting
#         plt.figure(figsize=(12, 6))
#         grouped_counts.T.plot(kind='bar', stacked=True, zorder=3, ax=plt.gca())
#         plt.title('Facebook Posts Distribution by Hour of the Day (09:00-17:00 Dominant Pages)')
#         plt.xlabel('Hour of the Day')
#         plt.ylabel('Number of Posts')
#         plt.xticks(range(24), [f'{hour:02d}:00' for hour in range(24)], rotation=45)
#         plt.legend(title='Page Name')
#         plt.grid(True, zorder=0)
#         st.pyplot(plt)
#     else:
#         st.write("Please upload a CSV file.")

# if st.button('Analyse Hourly Post Distribution'):
#     if uploaded_csv is not None:
#         # Read CSV file
#         df = pd.read_csv(uploaded_csv).copy()
#         # Parse the "Post Created Time" and extract hour
#         df['Hour'] = pd.to_datetime(df['Post Created Time'], format='%H:%M:%S').dt.hour

#         # Filter hours from 09 to 17
#         target_hours = df[df['Hour'].between(9, 17)]

#         # Group by page name and hour, then count posts
#         total_counts = df.groupby('User Name').size()
#         target_counts = target_hours.groupby('User Name').size()

#         # Calculate the percentage of posts in target hours
#         percentage = (target_counts / total_counts) * 100

#         # Filter pages with at least 90% of posts in target hours
#         target_pages = percentage[percentage >= 90].index

#         # Check if there are any pages that meet the criteria
#         if not target_pages.empty:
#             # Filter the original dataframe for the target pages
#             filtered_df = df[df['User Name'].isin(target_pages)]

#             # Group by page name and hour, then count posts for filtered pages
#             grouped_counts = filtered_df.groupby(['User Name', 'Hour']).size().unstack(fill_value=0)

#             # Check if there is data to plot
#             if not grouped_counts.empty:
#                 # Plotting
#                 plt.figure(figsize=(12, 6))
#                 grouped_counts.T.plot(kind='bar', stacked=True, zorder=3, ax=plt.gca())
#                 plt.title('Facebook Posts Distribution by Hour of the Day (09:00-17:00 Dominant Pages)')
#                 plt.xlabel('Hour of the Day')
#                 plt.ylabel('Number of Posts')
#                 plt.xticks(range(24), [f'{hour:02d}:00' for hour in range(24)], rotation=45)
#                 plt.legend(title='User Name')
#                 plt.grid(True, zorder=0)
#                 st.pyplot(plt)
#             else:
#                 st.write("No data available for plotting after filtering.")
#         else:
#             st.write("No pages meet the specified criteria.")
#     else:
#         st.write("Please upload a CSV file.")


# Upload CSV
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("Posting Cycle")

uploaded_csv = st.file_uploader("Upload CSV", type=['csv'])

# Allow the user to select the start and end hour
start_hour, end_hour = st.slider(
    'Select the time range:',
    0, 23, (9, 17)
)

# Display the selected time range
st.write(f"Selected time range: {start_hour:02d}:00 to {end_hour:02d}:00")

# Button to start the analysis
if st.button('Check Matching Pages'):
    if uploaded_csv is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_csv).copy()
        # Parse the "Post Created Time" and extract hour
        df['Hour'] = pd.to_datetime(df['Post Created Time'], format='%H:%M:%S').dt.hour

        # Filter hours based on selection
        target_hours = df[df['Hour'].between(start_hour, end_hour)]

        # Group by page name and hour, then count posts
        total_counts = df.groupby('User Name').size()
        target_counts = target_hours.groupby('User Name').size()

        # Calculate the percentage of posts in target hours
        percentage = (target_counts / total_counts) * 100

        # Filter pages with at least 90% of posts in target hours
        target_pages = percentage[percentage >= 90].index

        # Check if there are any pages that meet the criteria
        if not target_pages.empty:
            # Filter the original dataframe for the target pages
            filtered_df = df[df['User Name'].isin(target_pages)]

            # Group by page name and hour, then count posts for filtered pages
            grouped_counts = filtered_df.groupby(['User Name', 'Hour']).size().unstack(fill_value=0)

            # Check if there is data to plot
            if not grouped_counts.empty:
                # Plotting
                plt.figure(figsize=(12, 6))
                grouped_counts.T.plot(kind='bar', stacked=True, zorder=3, ax=plt.gca())
                plt.title(f'Facebook Posts Distribution by Hour of the Day ({start_hour:02d}:00-{end_hour:02d}:00 Dominant Pages)')
                plt.xlabel('Hour of the Day')
                plt.ylabel('Number of Posts')
                plt.xticks(range(24), [f'{hour:02d}:00' for hour in range(24)], rotation=45)
                plt.legend(title='User Name')
                plt.grid(True, zorder=0)
                st.pyplot(plt)
            else:
                st.write("No data available for plotting after filtering.")
        else:
            st.write("No pages meet the specified criteria.")
    else:
        st.write("Please upload a CSV file.")
