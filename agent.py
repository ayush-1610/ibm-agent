from crewai import Crew, Task, Agent
from crewai import SerperDevTool
from langchain_ibm import WastonxLLM
import os


from dotenv import load_dotenv
load_dotenv()

os.environ["WASTONX_API_KEY"]
os.environ["SERPER_API_KEY"]

parameter = {"decoding_method":"greedy", "max_new_tokens":500}
llm=WatsonxLLM(
    model_id="mets-llama/llama-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    params="parameters",
    project_id="c5b4b29a-0fcb-462e-a0a3-51a00d725885",

)

parameter = {"decoding_method":"greedy", "max_new_tokens":500}
function_calling_llm=WatsonxLLM(
    model_id="ibm-mistral/merlinite-7b",
    url="https://us-south.ml.cloud.ibm.com",
    params="parameters",
    project_id="c5b4b29a-0fcb-462e-a0a3-51a00d725885",
)

# Tools
search = SerperTool()

# Agent1
researcher = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="Senior AI Researcher",
    goal="Find promising research in the fields of quantum computing",
    backstory="You are a vetran quantum computing researcher with a background in mordern physics",
    allow_delegation="False",
    tools=[search],
    verbose=1
)

# Task
task1= Task(
    description="Search the internet andfind couple of examples of promising AI reaserch",
    expected_output="A detailed bullet point summary on each of the points. Each bullet point shlound cover the topic, backgroung and why the innovation is useful",
    output_file="task1output.txt",
    agent=researcher
)

# Agent2
writer = Agent(
    llm=llm,
    role="Senior Speech Writer",
    goal="Write engaging and witty keynote speeches from provided research",
    backstory="You are a vetran quantum computing writer with a background in mordern physics",
    allow_delegation="False",
    verbose=1
)

# Task
task2= Task(
    description="Write an engaging keynote speech on quantum computing",
    expected_output="A detailed keynote apeech with an intro, body and conclusion",
    output_file="task2output.txt",
    agent=researcher
)

# Crew
crew = Crew(agent=[researcher,writer], task=[task1, task2], verbose=1)
print(crew.kickoff())
