import streamlit as st
import pandas as pd
import re
import openai
from tenacity import retry, stop_after_attempt, wait_fixed
from streamlit_option_menu import option_menu
import os

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MAX_TOKENS = 2048

st.set_page_config(
    page_title="Chat Editor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = openai.OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz",
)

system_prompt = (
    "I will give you an answer, fix it. First, reason about the errors and the improvements "
    "you would make, based on the instruction. Then, fix the errors and improve the answer.\n\n"
    "Please use correct grammar, punctuation, and capitalization. If you find that the used "
    "lexicon is not appropriate, please change it to something more suitable.\n"
    "Do the best you can to make the answer sound natural and fluent. Always use $$ for block "
    "math and $ for inline math, and write using GitHub Markdown syntax."
    "Always answer in Italian.\n\n"
    "Do the best fix possible, as you will be evaluated on the quality of your work.\n\n"
    "If you are not sure about the answer, please leave it as it is.{user_messages}\n\n"
    "Here is the question being answered:\n{question}\n\n"
    "Instruction to follow while editing:\n{instruction}\n\n"
    "You should use the following format:\n"
    "```markdown\n"
    "## Reasoning\n"
    "[reasoning on errors and possible improvements]\n\n"
    "## Improved answer\n"
    "[improved answer in Italian]\n"
    "```"
)


def extract_answer_response(text: str):
    sections = text.split("# Improved answer")
    if len(sections) > 1:
        return sections[1].strip().strip("```").strip()
    else:
        return text.strip()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def complete(question: str, answer: str, conversation, instruction):
    previous_messages = [
        message["content"] for message in conversation if message["role"] == "user"
    ][:-1][:3]
    if len(previous_messages) == 0:
        user_messages = ""
    else:
        user_messages = "\n---\n".join(previous_messages) + "\n---"
        user_messages = """\nHere I will give you the previous user messages as context:\n{user_messages}"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt.format(
                    question=question,
                    user_messages=user_messages,
                    instruction=instruction,
                ),
            },
            {
                "role": "user",
                "content": answer.strip(),
            },
        ],
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
    )

    result = chat_completion.choices[0].message.content
    return extract_answer_response(result)


def convert_latex(latex_string):
    # replace \( and \) with $
    latex_string = re.sub(r"\\\(", "$", latex_string)
    latex_string = re.sub(r"\\\)", "$", latex_string)

    # replace \[ and \] with $$
    latex_string = re.sub(r"\\\]", "$$", latex_string)
    latex_string = re.sub(r"\\\[", "$$", latex_string)

    return latex_string


def cleaning(text: str):
    text = re.sub(r" +", " ", text)
    # double new lines
    text = re.sub(r"\n\n+", "\n\n", text)

    # convert E' to È
    text = re.sub(r"E'", "È", text)
    text = re.sub(r"e'", "è", text)

    # convert every type of quote to '
    text = re.sub(r"''", '"', text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"‘", "'", text)
    text = re.sub(r"“", '"', text)
    text = re.sub(r"”", '"', text)
    text = re.sub(r"«", '"', text)
    text = re.sub(r"»", '"', text)

    # convert every type of dash to -
    text = re.sub(r"—", "-", text)
    text = re.sub(r"–", "-", text)
    text = re.sub(r"—", "-", text)
    text = re.sub(r"‐", "-", text)

    return text


@st.cache_data
def load_data(path):
    data = pd.read_json(
        path,
        lines=True,
        orient="records",
    )
    return data


if "df" not in st.session_state:
    st.markdown(
        """
        <style>
        .gradient-text {
            font-size: 40px;
            background: -webkit-linear-gradient(45deg, #00FFFF, #00BFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-family: "Roboto", sans-serif;
        }
        </style>
        <div class="gradient-text" style="text-align: center">Hello, welcome to the Editor!</div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    menu = ["Load from path", "Upload file"]
    choice = option_menu("Select an option", menu, 0)
    st.info(
        """
        Here you can load the chat data. You can either load it from a path or upload it.
        
        Remember: the file must be in JSON format and must have a column `messages` with the following structure:
        ```
        [
            {
                "role": "system/user/assistant",
                "content": "..."
            },
            ...
        ]
        ```
        """
    )
    if choice == "Load from path":
        add_file = st.text_input(
            "File path",
            value="conversations.json",
        )
        if st.button("Load"):
            try:
                st.session_state.df = load_data(add_file)
                st.success("Loaded!")
                st.rerun()
            except:
                st.error("File not found, please try again")
    elif choice == "Upload file":
        uploaded_file = st.file_uploader("Choose a file")
        if st.button("Load"):
            try:
                st.session_state.df = load_data(uploaded_file)
                st.success("Loaded!")
                st.rerun()
            except:
                st.error("File not found, please try again")
else:
    chat = st.empty()
    with st.sidebar:
        if "row" not in st.session_state:
            st.session_state.row = 0

        st.markdown(
            """
            <style>
            .gradient-text {
                font-size: 30px;
                background: -webkit-linear-gradient(45deg, #00FFFF, #00BFFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: bold; <!-- use Roboto font -->
                font-family: "Roboto", sans-serif;
            }
            </style>
            <div class="gradient-text" style="text-align: center">Editor</div>
            """,
            unsafe_allow_html=True,
        )

        # add a space
        st.write("")
        st.info(
            "Here you can edit the chat data.\n\nInstructions:\n\n1. Select a row from the slider or the next and previous buttons.\n2. Edit the messages with the available tools.\n3. Save the file with the save button."
        )
        st.markdown("### Select row")

        # next and previous buttons in a container
        with st.container():
            col1, col2 = st.columns(2)
            previous_button = col1.button("Previous", key="previous")
            next_button = col2.button("Next", key="next    ")

        if next_button:
            st.session_state.row = min(
                st.session_state.row + 1, len(st.session_state.df) - 1
            )
        elif previous_button:
            st.session_state.row = max(st.session_state.row - 1, 0)

        if "row" not in st.session_state:
            st.session_state.row = st.slider(
                "Select row",
                0,
                len(st.session_state.df) - 1,
                st.session_state.row,
                key="row",
            )

        st.markdown(f"## Row {st.session_state.row}")

        # adds jump to row
        jump_to_index = st.number_input(
            "Jump to row", 0, len(st.session_state.df) - 1, 0
        )
        if st.button("Jump") and jump_to_index < len(st.session_state.df):
            st.session_state.row = jump_to_index

        st.markdown("### Save")

        file_name = st.text_input("File name", value="conversations.json")

        save_button = st.button("Save", key="save")
        if save_button:
            st.success("Saved!")
            st.session_state.df.to_json(file_name, orient="records", lines=True)

    for i, roww in enumerate(
        st.session_state.df.iloc[st.session_state.row]["messages"]
    ):
        role = roww["role"]
        message = st.chat_message(role)
        message.markdown(roww["content"])

        with message.expander("Edit"):
            new_message = st.text_area("Edit message", value=roww["content"])
            col1, col2, col3, col4 = st.columns(4)
            if col1.button(f"Update {i}"):
                message.markdown(new_message)
                st.session_state.df.iloc[st.session_state.row]["messages"][i][
                    "content"
                ] = new_message

            if col2.button(f"Format LaTeX {i}"):
                message.markdown(convert_latex(new_message))
                st.session_state.df.iloc[st.session_state.row]["messages"][i][
                    "content"
                ] = convert_latex(new_message)

            if col3.button(f"Clean {i}"):
                message.markdown(cleaning(new_message))
                st.session_state.df.iloc[st.session_state.row]["messages"][i][
                    "content"
                ] = cleaning(new_message)

            if st.session_state.df.iloc[st.session_state.row]["messages"][i - 1] and (
                st.session_state.df.iloc[st.session_state.row]["messages"][i - 1][
                    "role"
                ]
                == "user"
            ):
                instruction = st.text_input(
                    f"Instruction {i}",
                    value="Migliora la formattazione e la grammatica.",
                )
                if col4.button(f"Rewrite {i}"):
                    new_message = complete(
                        st.session_state.df.iloc[st.session_state.row]["messages"][
                            i - 1
                        ]["content"],
                        new_message,
                        st.session_state.df.iloc[st.session_state.row]["messages"],
                        instruction,
                    )
                    message.markdown(new_message)
                    st.session_state.df.iloc[st.session_state.row]["messages"][i][
                        "content"
                    ] = new_message
