import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import json

load_dotenv()
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Smart Router API is running", "version": "1.0.0"}

api_registry = [
    {
        "intent": "submit_timesheet",
        "name": "PayrollX SubmitTimesheet",
        "description": "Submits a weekly timesheet with work hours and employee ID",
        "endpoint": "https://example.com/api/timesheet",
        "schema": {
            "date": "string",
            "hours": "float",
            "employee_id": "string"
        }
    },
    {
        "intent": "leave_request",
        "name": "HR Leave API",
        "description": "Creates a leave request for an employee with start and end date",
        "endpoint": "https://example.com/api/leave",
        "schema": {
            "start_date": "string",
            "end_date": "string",
            "employee_id": "string",
            "type": "string"
        }
    }
]

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

@tool
def classify_intent(input_data: str) -> str:
    """
    Classify the user's input into one of the predefined intent categories.
    
    This tool analyzes the natural language input and determines if the user wants to:
    - submit a timesheet
    - request leave
    - claim expenses
    
    Args:
        input_data: The user's natural language request
        
    Returns:
        A string representing the classified intent
    """
    prompt = PromptTemplate.from_template("""
You are an intent classification assistant. Based on the user's input, classify it as one of the following intents:

submit_timesheet, leave_request, expense_claim

User input:
{input_data}

Return only the intent name.
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input_data=input_data)
    return result.strip().lower()

@tool
def select_api(intent: str) -> dict:
    """
    Select the most appropriate API from the known registry based on the provided user intent.

    The function compares the user's intent against a list of known APIs and their descriptions,
    and uses an LLM to determine the best match. It returns the metadata of the selected API,
    including its name, endpoint, and expected JSON schema.
    """
    options_text = "\n".join([
        f"{i+1}. {api['name']} - {api['description']}" for i, api in enumerate(api_registry)
    ])
    prompt = PromptTemplate.from_template("""
You are an API selection assistant. Given a user intent and a list of available APIs, select the best match by number.

Intent: {intent}

Available APIs:
{options_text}

Respond only with the number of the best API to use.
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(intent=intent, options_text=options_text).strip()
    try:
        selected_index = int(result) - 1
        return api_registry[selected_index]
    except:
        return {"error": "API selection failed"}

@tool
def transform_json(raw_input: str, schema: dict) -> dict:
    """
    Transform the raw user input into the format expected by the selected API schema.
    
    This function takes natural language input and converts it to structured JSON
    that matches the required schema for the API call.
    
    Args:
        raw_input: The user's request in natural language
        schema: The JSON schema definition required by the API
        
    Returns:
        A dictionary containing the properly formatted data for the API call
    """
    # Create a prompt for the LLM to extract structured data
    prompt = PromptTemplate.from_template("""
You are a data extraction assistant. Extract the relevant information from the user input and format it according to the provided schema.

User input:
{raw_input}

Required schema:
{schema}

Return ONLY a valid JSON object with the extracted information.
""")
    
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(raw_input=raw_input, schema=schema)
    
    try:
        # Clean the result in case the LLM included backticks or other formatting
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except:
        # Fallback to a simple example if JSON parsing fails
        return {"example": "transformed data"}

@tool
def call_api(endpoint: str, payload: dict) -> dict:
    """
    Call the selected API endpoint with the JSON payload.
    
    This function makes an HTTP POST request to the specified endpoint with the 
    provided payload data. It handles the API communication and returns the 
    response status and body.
    
    Args:
        endpoint: The URL of the API endpoint to call
        payload: The JSON payload to send to the API
        
    Returns:
        A dictionary containing the API response status and body
    """
    try:
        # For development purposes, we'll simulate the API call
        # In production, uncomment the actual API call code
        # response = requests.post(endpoint, json=payload)
        # return {"status": response.status_code, "body": response.json()}
        
        # Simulation response
        return {
            "status": 200, 
            "body": {
                "success": True, 
                "message": f"Successfully processed request to {endpoint}",
                "data": payload
            }
        }
    except Exception as e:
        return {"error": str(e)}

llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_type="azure",
    temperature=0
)

tools = [
    Tool.from_function(
        func=classify_intent,
        name="classify_intent",
        description="Classify the user's input into one of the predefined intent categories (submit_timesheet, leave_request, expense_claim)"
    ),
    Tool.from_function(
        func=select_api,
        name="select_api",
        description="Select the most appropriate API from the known registry based on the provided user intent"
    ),
    Tool.from_function(
        func=transform_json,
        name="transform_json",
        description="Transform the raw user input into the format expected by the selected API schema"
    ),
    Tool.from_function(
        func=call_api,
        name="call_api",
        description="Call the selected API endpoint with the JSON payload"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

class InputPayload(BaseModel):
    input: str

@app.post("/smart-router")
async def smart_router(payload: InputPayload):
    try:
        result = agent.run(payload.input)
        return JSONResponse(content={"result": result, "success": True})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )