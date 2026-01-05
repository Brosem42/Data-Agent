# importing libraries 
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel 
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openai import OpenAIChatCompletion
from typing import Annotated
import asyncio
from datetime import datetime 
from dataclasses import dataclass, field
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from contextlib import redirect_stdout

load_dotenv()

#init model
model = OpenAIChatModel('gpt-4.1', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))

#define state--> basically just defining what is happening between my agent and the user
@dataclass 
class State:
    user_query: str = field(default_factory=str)
    file_name: str = field(default_factory=str)

#defining the tools my agent will use to perform specific actions within data analysis

#Tool 1: defining tool for reading csv files
def get_column_list(
        file_name: Annotated[str, "The file that has the data"]):
    df = pd.read_csv(file_name)
    columns = df.columns.tolist()
    return str(columns)

##Tool 2: defining tool to get the description of columns
def get_column_description(
        column_dict: Annotated[dict, "Dictionary of columns + description of column"]):
    
    return str(column_dict)

#Tool 3: defining tool for generating my graph of data collected
def generate_graph(
        code: Annotated[str, "Code to generate the visualizations"]) -> str:
    catcher = StringIO()
    
    try:
        with redirect_stdout(catcher):
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, globals(), globals())
            return(
                f"Graph path is in \n\n{catcher.getvalue()}\n"
                f"Once successful, you may proceed to the next step."
            )
    except Exception as e:
        return f"Failed to run code. Error: {repr(e)}, try different pathway approach"
    
#Tool 4: tool that executes the python code for metrics of each variable item, object
def python_execution_tool(
        code: Annotated[str, "Python code generated for data calculation and processing."]) -> str:
    catcher = StringIO()

    try:
        with redirect_stdout(catcher):
            #observe for syntax errors early on
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, globals, globals())

            return (
                f"The metric value is at \n\n{catcher.getvalue}\n"
                f"Once successful, you may proceed to the next step and include this value in the report."
            )
    except Exception as e:
        return f"Failed to run code. Error: {repr(e)}, try different pathway approach"
    

# Create and invoke analyst agent + connect my tools
class DataAgentOutput(BaseModel):
    analysis_report: str = Field(description="The analysis report of the data is in markdown format.")
    metrics: list[str] = Field(description="The metrics of the data.")
    image_html_path: str = Field(description="The confirmed path of the graph in html format. If there is no graph generated, return empty string.")
    image_png_path: str = Field(description="The confirmed path of the graph in png format. If there is no graph generated, return empty string.")
    conclusion: str = Field(description="The final analysis concludes here.")

#invoking custom made tools
data_agent = Agent(
    model=model,
    tools=[Tool(get_column_list, takes_ctx=False), 
           Tool(get_column_description, takes_ctx=False), 
           Tool(generate_graph, takes_ctx=False), 
           Tool(python_execution_tool, takes_ctx=False)],
           deps_type=State,
           result_type=DataAgentOutput,
           instrument=True
)

#system prompting --> object 
@data_agent.system_prompt
async def get_data_agent_system_prompt(ctx: RunContext[State]):
    prompt = f"""
    You are an expert data analyst agent who has successfully mastered the task to analyze the data provided by the user. 
    You are repsonsible for executing comprehensive, but user-friendly data analysis workflows and generating professional analytic reports for both technical and non-technical stakeholders, 
    users with or without AI or machine learning experience, and lastly for technical data engineers and machine learning engineers who work in the field.

    **Your Task:"**
    Analyze the provided dataset to answer the user's query thorugh systematic data exploration, statistical analysis, and visualization.
    Deliver actionable insights through a comprehensive report backed by quantitative evidence.

    **Tools Available to you for use:"
    - `get_column_list`: Retrieve all columns names from the dataset.
    - `get_column_description`: Get detailed description and metadata for specific columns.
    - `generate_graph`: Generate visualizations that are customizable for the user (charts, plots, graphs) and save them in HTML and PNG formats. Use Plotly express library to make the graph interactive for the users.
    - `python_execution_tool`: Execute Python code for statistical calculations, data processing, and metric computation.

    **Context (Input):**
    - User Query: {ctx.deps.user_query}
    - Dataset File Name: {ctx.deps.file_name}

    **Execution Workflow:**
    **HIGHLY CRITICAL**: The state is not persisent between tool calls. Make sure to always reload the dataset and import necessary libraries in each Python execution.

    1. **Dataset Discovery**: Use `get_column_list` to retrieve all available columns, then use `get_column_description` to understand column meanings and the data types.
    
    2. **Analysis Planning**: Based on user query and dataset structure, create a systematic analysis plan identifying the following:
    - Key variables to examine
    - Statistical methods to apply
    - Visualizations to create
    - Metrics to calculate

    3. Data Exploration**: Load the datast using pandas and perform initial exploration:
    - Check data shape, types, and quality
    - Identify missing values and outliers
    - Generate descriptive statistics 

    4. **Statistical Analysis**: This is where you will execute the planned analysis using appropiate statistical methods by performing the following:
    - Calculate relevant metrics and aggregations
    - Perform hypothesis testing if applicable
    - Identify patterns, trends, and correlations

    5. **Vizualization Creation**: 
"""