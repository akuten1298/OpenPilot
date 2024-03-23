import os 
import sys
from langchain_community.tools import ShellTool
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) > 1:
    repo_link = sys.argv[1]
    print(f"Input argument received: {repo_link}")
else:
    print("Please pass a repository link")

def executeAgent():
    shell_tool = ShellTool()

    llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview")

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")
    self_ask_with_search = initialize_agent(
        [shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
        handle_parsing_errors=True,
        memory=conversational_memory,
    )
    self_ask_with_search.run(
        f"""You are an AI Agent assistant to help user with codebase setup.
        You are supposed to specifically work on all things related to setting up the repository with respect to Git.
        
        REMEMBER: 
        The instruction below are the sequence of steps and are all relative to each other. Perform one after the other and if any step fails, 
        auto fix and keep retrying until it succeeds.

        REMEMBER: I will not manually execute anything but you can do it for me. Do not ask for my intervention, just do all the steps by yourself.

        REMEMBER: NEVER DELETE ANY FUCKIN THING.

        If there is any issue with the current step then go to the next step.

        STEPS:


        1. This is the github repo you are required to work with: {repo_link}
        
        3. create a folder inside the current directory with the repo name and then cd into this folder. Do not check if it already exists, just cd into this folder.
        Example cd command: cd <repo_name>
        

        4. Now use this new repo name folder and clone the project directly under this folder.
        Example clone command: git clone {repo_link}

        5. Check if pip exists and if not install pip
        8. if requirements.txt does exist then Display the contents of the file. From requirements.txt, extract the package names and versions and execute the pip install commands for all of the packages
        11. Now open a new Visual Studio code with the cloned repo
        13. Scan the README or readme file inside this repository and then list the steps of commands required for installation and execution seperately.
        14. First run the installation steps specific to the system that you are running on.
        15. Once installation step is done proceed to run the repository project or the steps required to execute an example of the project.
        14. Once you get the list of commands, execute them.
        """

    )

executeAgent()
