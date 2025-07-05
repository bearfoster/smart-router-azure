import os
from typing import Any, Dict, List, Optional, Union
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
        "endpoint": "http://localhost:3002/api/timesheet/",
        "method": "POST",
        "schema": {
            "employee_id": "string",
            "data": "timesheet_data"
        }
    },
    {
        "intent": "leave_request",
        "name": "HR Leave API",
        "description": "Creates a leave request for an employee with start and end date",
        "endpoint": "http://localhost:3002/api/leave/",
        "method": "POST",
        "schema": {
            "employee_id": "string",
            "data": "leave_request_data"
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
    print(f"[INFO] classify_intent: Processing user input: {input_data[:50]}...")
    result = chain.run(input_data=input_data)
    intent = result.strip().lower()
    print(f"[INFO] classify_intent: Classified intent: {intent}")
    return intent

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
    print(f"[INFO] select_api: Processing intent={intent}")
    result = chain.run(intent=intent, options_text=options_text).strip()
    try:
        selected_index = int(result) - 1
        print(f"[INFO] select_api: Selected API index {selected_index}, name={api_registry[selected_index]['name']}")
        return api_registry[selected_index]
    except Exception as e:
        print(f"[ERROR] select_api: Error selecting API: {str(e)}")
        # Return a more descriptive error and include the intent
        return {
            "error": f"API selection failed for intent: {intent}",
            "details": str(e),
            "available_apis": [api["name"] for api in api_registry]
        }

@tool
def transform_json(raw_input: str, api_details: dict = None, schema: dict = None) -> dict:
    """
    Transform the raw user input into the format expected by the selected API schema.
    
    This function takes natural language input and converts it to structured JSON
    that matches the required schema for the API call. It also preserves the API endpoint details.
    
    Args:
        raw_input: The user's request in natural language
        api_details: The details of the selected API including endpoint (optional)
        schema: The JSON schema definition required by the API (optional)
        
    Returns:
        A dictionary containing the properly formatted data for the API call and API endpoint
    """
    print(f"[INFO] transform_json: raw_input={raw_input[:50]}..., api_details={api_details}")
    
    # Extract schema from api_details if provided
    if api_details and isinstance(api_details, dict) and 'schema' in api_details:
        schema = api_details.get('schema')
        print(f"[INFO] transform_json: Using schema from provided API details")
    # Use a default schema if none is provided
    elif schema is None:
        # Try to use schema from the first API that matches
        api_to_use = None
        for api in api_registry:
            if 'timesheet' in raw_input.lower() and 'timesheet' in api['intent'].lower():
                api_to_use = api
                break
            elif 'leave' in raw_input.lower() and 'leave' in api['intent'].lower():
                api_to_use = api
                break
        
        if api_to_use:
            schema = api_to_use['schema']
            print(f"[INFO] transform_json: Using schema from {api_to_use['name']}")
        else:
            schema = {"example": "field"}
            print(f"[INFO] transform_json: No matching API found, using default schema")
    
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
        parsed_result = json.loads(result)
        print(f"[INFO] transform_json: Successfully parsed result")
        
        # Create a result that includes both the transformed data and API endpoint
        if api_details and isinstance(api_details, dict) and 'endpoint' in api_details:
            # Add metadata to help call_api function
            return {
                "payload": parsed_result,
                "endpoint": api_details.get('endpoint'),
                "api_name": api_details.get('name', 'Unknown API')
            }
        return parsed_result
    except Exception as e:
        # Fallback to a simple example if JSON parsing fails
        print(f"[ERROR] transform_json: Error parsing result: {str(e)}")
        return {"example": "transformed data"}

@tool
def call_api(api_data: Any = None, endpoint: str = None, payload: dict = None) -> dict:
    """
    Call the selected API endpoint with the JSON payload.
    
    This function makes an HTTP POST request to the specified endpoint with the 
    provided payload data. It handles the API communication and returns the 
    response status and body.
    
    Args:
        api_data: Combined API endpoint and payload information (from transform_json)
        endpoint: The URL of the API endpoint to call (optional if api_data provided)
        payload: The JSON payload to send to the API (optional if api_data provided)
        
    Returns:
        A dictionary containing the API response status and body
    """
    print(f"[INFO] call_api: Initial args - api_data type: {type(api_data)}, value: {api_data}")
    
    # Handle case where api_data is passed as a string (from agent)
    if isinstance(api_data, str):
        try:
            # Try to parse it as JSON
            print(f"[INFO] call_api: api_data is a string, attempting to parse as JSON")
            parsed_data = json.loads(api_data)
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                # If it's a list, use the first item as the payload
                print(f"[INFO] call_api: api_data parsed as a list, using first item as payload")
                payload = parsed_data[0]
                endpoint = api_registry[0]['endpoint'] if api_registry else "http://localhost:3002/api/timesheet"
            elif isinstance(parsed_data, dict):
                # If it's a dict, use it directly
                api_data = parsed_data
                print(f"[INFO] call_api: Parsed string into dictionary: {api_data}")
            else:
                print(f"[WARNING] call_api: Could not use parsed string data: {parsed_data}")
                return {"error": f"Invalid api_data format after parsing: {str(parsed_data)[:100]}..."}
        except json.JSONDecodeError as e:
            print(f"[ERROR] call_api: Could not parse api_data as JSON: {str(e)}")
            return {"error": f"Invalid JSON format in api_data: {api_data[:100]}..."}
    
    # Handle combined api_data (from transform_json)
    if api_data and isinstance(api_data, dict):
        if 'endpoint' in api_data and 'payload' in api_data:
            endpoint = api_data.get('endpoint')
            payload = api_data.get('payload')
            print(f"[INFO] call_api: Extracted endpoint '{endpoint}' and payload from api_data")
        elif endpoint is None and payload is None:
            # api_data itself might be the payload
            if 'endpoint' not in api_data:
                # Use the first API endpoint by default
                endpoint = api_registry[0]['endpoint'] if api_registry else "http://localhost:3002/api/timesheet"
                payload = api_data
                print(f"[INFO] call_api: Using default endpoint={endpoint} with api_data as payload")
    
    print(f"[INFO] call_api: Final endpoint={endpoint}, payload={payload}")
    
    try:
        # Make the API call
        print(f"[INFO] call_api: Making a real API call to {endpoint}")
        response = requests.post(endpoint, json=payload, timeout=10)
        print(f"[INFO] call_api: Response status code: {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"[INFO] call_api: Response content: {response_json}")
            return {"status": response.status_code, "body": response_json}
        except ValueError:
            # Handle non-JSON responses
            print(f"[WARNING] call_api: Non-JSON response: {response.text[:100]}...")
            return {"status": response.status_code, "body": {"raw_response": response.text}}
    except Exception as e:
        print(f"[ERROR] call_api: Error: {str(e)}")
        return {"error": str(e), "endpoint": endpoint, "payload": payload}

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
        description="Classify the user's input into one of the predefined intent categories (submit_timesheet, leave_request, expense_claim). Returns a string with the intent name."
    ),
    Tool.from_function(
        func=select_api,
        name="select_api",
        description="Select the most appropriate API from the known registry based on the provided user intent. Input: intent string from classify_intent. Returns: API details including endpoint and schema."
    ),
    Tool.from_function(
        func=transform_json,
        name="transform_json",
        description="Transform the raw user input into a JSON payload that matches the API schema. Input: raw_input (string) and api_details (dict from select_api). Returns: dict with 'payload' and 'endpoint' fields."
    ),
    Tool.from_function(
        func=call_api,
        name="call_api",
        description="Call the selected API endpoint with the JSON payload. Input: api_data (from transform_json) or endpoint and payload separately. Returns: API response with status and body."
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
        error_msg = str(e)
        import traceback
        stack_trace = traceback.format_exc()
        print(f"[ERROR] Exception in smart-router endpoint: {error_msg}")
        print(f"[ERROR] Stack trace:\n{stack_trace}")
        
        # Return a more detailed error response
        return JSONResponse(
            status_code=500,
            content={
                "error": error_msg,
                "success": False,
                "trace": stack_trace.split('\n')
            }
        )