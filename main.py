import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import json

load_dotenv()
app = FastAPI()

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
    return {"example": "transformed data"}

@tool
def call_api(endpoint: str, payload: dict) -> dict:
    try:
        response = requests.post(endpoint, json=payload)
        return {"status": response.status_code, "body": response.json()}
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
    Tool.from_function(classify_intent),
    Tool.from_function(select_api),
    Tool.from_function(transform_json),
    Tool.from_function(call_api)
]

agent = initialize_agent(
    tools,
    llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

class InputPayload(BaseModel):
    input: str

@app.post("/smart-router")
async def smart_router(payload: InputPayload):
    result = agent.run(payload.input)
    return JSONResponse(content={"result": result})