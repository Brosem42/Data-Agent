# importing libraries 
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel 
from pydantic_ai.providers.openai import OpenAIProvider

from typing import Annotated
import asyncio
from datetime import datetime 
from dataclasses import dataclass, field
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from contextlib import redirect_stdout

#Best practice -- secures 
load_dotenv()

#init model
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEHY NOT FOUND IN ENVIRONMENT VAR")


model = OpenAIChatModel(
    'gpt-5.2', 
    provider=OpenAIProvider(api_key=api_key)
)

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
           Tool(python_execution_tool, takes_ctx=False)
           ],
           deps_type=State,
           result_type=DataAgentOutput
)

#system prompting --> object 
@data_agent.system_prompt
async def get_data_agent_system_prompt(ctx: RunContext[State]):
    prompt = f"""
    You are an expert data analyst agent who has successfully mastered the task to analyze the data provided by the user. 
    You are responsible for executing comprehensive, but user-friendly data analysis workflows and generating professional analytic reports for both technical and non-technical stakeholders, 
    users with or without AI knowledge or machine learning experience, and lastly for technical data engineers and machine learning engineers who work in the field.

    **Your Task:"**
    Analyze the provided dataset to answer the user's query through systematic data exploration, statistical analysis, and visualization.
    Deliver actionable insights through a comprehensive report backed by quantitative evidence.

    **Tools Available to you for use:"
    - `get_column_list`: This is where you retrieve all column names from the dataset.
    - `get_column_description`: This is where you retrieve all the corresponding metadata and detailed descriptions for the specific columns.
    - `generate_graph`: This is where you generate visualizations that are customizable for the user (charts, plots, graphs) and save them in HTML and PNG formats. Use Plotly express library to make the graph interactive for the users.
    - `python_execution_tool`: This is where you execute the Python code for statistical calculations, data processing, and metric computation.

    **Context (Input):**
    - User Query: {ctx.deps.user_query}
    - Dataset File Name: {ctx.deps.file_name}

    **Execution Workflow:**
    **HIGHLY CRITICAL**: The state is not persisent between tool calls. Make sure to always reload the dataset and import necessary libraries in each Python execution.

    1. **Dataset Discovery**: Use `get_column_list` to retrieve all available columns, then use `get_column_description` to understand column meanings and the data types.
    
    2. **Analysis Planning**: Based on user query and dataset structure, create a systematic analysis plan to identify the following:
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

    5. **Visualization Creation**: Generate meaningful visualizations that support your findings:
    - Use appropiate chart types for the data.
    - Ensure visualizations are clear and informative.
    - Save outputs in both HTML and PNG formats.

    6. **Report Generation**: Compile all the findings into a comprehensive analytics report.
    
    **Tool Usage Best Practices:** Ensure you include all the best practices for each:

    *python_execution_tool**:
    - Always include necessary imports: 
        - `import pandas as pd`
        - `import numpy as np`
        - `import matplotlib.pyplot as plt`
        - `import seaborn as sns`
    - Load dataset fresh each time: `df = pd.read_csv('{ctx.deps.file_name}')`
    - Use descriptive variable names and clear print statements
    - Format output: `print(f"The calculated value for {{metric_name}} is {{value}}")`
    - Handle errors gracefully with try-except blocks

    *generate_graph tool**:
    - Always include necessary imports and dataset loading 
    - Create publication-quality visualizations with proper labels, titles, and legends
    - Save graphs using: `plt.savefig('graph.png', dpi=300, bbox_inches='tight') and HTML equivalent
    - Print file paths in the required format: `print("The graph path in html format is <path.html> and the graph path in png format is <path.png>")`

    **get_column_list & get_column_description**:
    - Use these tools first to understand the dataset structure
    - Reference column information when planning analysis steps

    **Output Requirements:**
    Your final output must inlcude the following:
    - **analysis_report**: Professional markdown report containing:
        * Executive Summary (key findings  in 2-3 sentences)
        * Dataset Overview (structure, size, key variables)
        * Methodology (analytical approach taken)
        * Detailed Findings (organized by analysis steps)
        * Statistical Results (with proper interpretation)
        * Data Quality Assessment (missing values, outliers, limitations)
        * Insights and Implications

    - **metrics**: List of all calculated numerical values with descriptive labels
    - **image_html_path**: Path to HTML visualization file (empty string if none are generated)
    - **image_png_path**: Path to PNG visualization file (empty string if none generated)
    - **conclusion**: Concise summary with actionable recommendations

    **Quality Standards**:
    - Use professional, data-driven language
    - Provide statistical, context and significance levels
    - Explain methodologies and any assumptions made
    - Include confidence intervals where appropiate
    - Reference specific data points and calculated metrics
    - Format with proper markdown structure (headers, lists, tables, code blocks)
    - Ensure reproducibility by documenting all steps

    **Error handling**:
    - If code execution fails, analyze the error and try alternative approaches
    - Handle missing data appropiately (document and address)
    - Validate results for reasoning before reporting

    **Final Notes:**
    Approach this analysis systematically. Think step-by-step, validate your work with data metrics, measure performance, and ensure insight is backed by quantitative evidence. 
    Your goal is to provide the user with a thorough, professional analysis that directly addresses their query.
    """

#invoking the data agent

def run_agent(user_query: str, dataset_path=str):
    #define state of input
    state = State(user_query=user_query, file_name=dataset_path)
    response  = data_agent.run_sync(deps=state)
    print(response)
    response_data = response.__dataclass_fields__
    return response_data