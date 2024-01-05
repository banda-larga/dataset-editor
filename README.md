# Chat / Instructions Editor

## Description

A simple streamlit app to edit chat messages (to create datasets). Useful for chatbots and similar applications, where you need to create a dataset of messages.
I haven't found anything similar (w. streamlit / simple), so I decided to create a banana implementation.

The cleaning functions have to be customized if you want to use it for languages other than Italian.

## How to use

1. Install the requirements: `pip install -r requirements.txt`
2. Export `TOGETHER_API_KEY` as environment variable: `export TOGETHER_API_KEY=<your_api_key>`
3. Run the app: `streamlit run app.py`
4. Upload a chat file (e.g. `chat.json`) and start editing!
