import os
from apikey import HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY
# import requests
import streamlit as st
from streamlit_chat import message
# import openai
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

# Initialize OpenAI API with temperature of 0.7
llm = OpenAI(temperature=0.7)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

def main():
    # Set page title
    st.title("AB Bot")

    # If messages not in session state, initialize with default message
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        memory.chat_memory.add_ai_message("How can I help you?")

    # Create form for input and button to submit
    with st.form("chat_input", clear_on_submit=True):
        # Create two columns, the first for user input and the second for the submit button
        a, b = st.columns([4, 1])
        # Create a text input for user to enter message
        user_input = a.text_input(
            label="Your message:",
            placeholder="What would you like to say?",
            label_visibility="collapsed",
        )
        # Create a submit button
        b.form_submit_button("Send", use_container_width=True)

    # Loop through messages and display them on the page
    for idx, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["role"] == "user", key=idx)

    # If user input exists, add it to messages and get response from OpenAI
    # if user_input:
    #     st.session_state.messages.append({"role": "user", "content": user_input})
    #     message(user_input, is_user=True)
    #     response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #     msg = response.choices[0].message
    #     st.session_state.messages.append(msg)
    #     message(msg.content)

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        message(user_input, is_user=True)
        response = chain.run(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response, is_user=False)
    
if __name__ == "__main__":
    main()