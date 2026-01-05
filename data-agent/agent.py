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
    

# Create and invoke analyst agent
class DataAgentOutput(BaseModel):
    analysis_report: str = Field(description="The analysis report of the data is in markdown format.")
    metrics: list[str] = Field(description="The metrics of the data.")
    image_html_path: str = Field(description="The confirmed path of the graph. If there is no grpah, return empty string")
    