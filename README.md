# Smart Router Azure

An intelligent API routing service that uses Azure OpenAI to process natural language requests and automatically route them to appropriate backend APIs. This service acts as a smart middleware layer that can understand user intent, select the right API, transform data formats, and execute API calls.

## Features

- **Natural Language Processing**: Understands user requests in plain English
- **Intent Classification**: Automatically classifies user input into predefined categories
- **API Selection**: Intelligently selects the most appropriate API from a registry
- **Data Transformation**: Converts natural language input into structured JSON payloads
- **Automatic API Routing**: Makes API calls to the selected endpoints
- **Azure OpenAI Integration**: Leverages Azure's OpenAI service for AI capabilities
- **LangChain Framework**: Uses LangChain agents for sophisticated reasoning chains

## Architecture

The Smart Router follows a multi-step process:

1. **Intent Classification** - Analyzes the user's natural language input
2. **API Selection** - Chooses the best matching API from the registry
3. **Data Transformation** - Converts input to the required JSON schema
4. **API Execution** - Makes the actual API call and returns results

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI service access
- Access to target APIs (e.g., mock-payroll-api)

## Installation

1. Navigate to the smart-router-azure directory:
   ```bash
   cd smart-router-azure
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Configure your Azure OpenAI credentials in the `.env` file:

```env
OPENAI_API_TYPE=azure
OPENAI_API_KEY=your_azure_openai_api_key
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=2025-01-01-preview
OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

## Running the Service

### Development Mode
```bash
uvicorn main:app --reload --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The service will start on port 8000 by default.

## API Endpoints

### Health Check
- **GET** `/` - Check service status

### Smart Router
- **POST** `/smart-router` - Process natural language requests and route to appropriate APIs

## Usage Examples

### Submit a Timesheet Request
```bash
curl -X POST http://localhost:8000/smart-router \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I worked 9 to 5 Monday to Wednesday, had Thursday off, and worked Friday until 3. My employee ID is EMP123."
  }'
```

### Submit a Leave Request
```bash
curl -X POST http://localhost:8000/smart-router \
  -H "Content-Type: application/json" \
  -d '{
    "input": "I need to request vacation leave from August 1st to August 5th for a family trip. My employee ID is EMP001."
  }'
```

### Response Format
```json
{
  "result": "API call successful with response: {...}",
  "success": true
}
```

## Supported Intents

The Smart Router currently supports the following intent categories:

- **submit_timesheet** - For timesheet submissions
- **leave_request** - For leave/vacation requests
- **expense_claim** - For expense reporting (extensible)

## API Registry

The service maintains a registry of available APIs:

```python
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
```

## How It Works

### 1. Intent Classification
The service uses Azure OpenAI to analyze natural language input and classify it into predefined categories.

### 2. API Selection
Based on the classified intent, the service selects the most appropriate API from its registry using LLM-powered matching.

### 3. Data Transformation
The service converts the natural language input into structured JSON that matches the target API's schema requirements.

### 4. API Execution
Finally, the service makes the actual HTTP request to the selected API endpoint with the transformed data.

## Dependencies

- **FastAPI** - Web framework for building APIs
- **Uvicorn** - ASGI server for FastAPI
- **LangChain** - Framework for building LLM applications
- **LangChain Community** - Community tools and integrations
- **OpenAI** - OpenAI Python client
- **Requests** - HTTP library for making API calls
- **Python-dotenv** - Environment variable management

## Error Handling

The service includes comprehensive error handling:

- Invalid input validation
- API selection failures
- Data transformation errors
- Target API communication errors
- Azure OpenAI service errors

Errors are returned with detailed information and stack traces in development mode.

## Extending the Service

### Adding New APIs
1. Add new API definitions to the `api_registry` list
2. Update intent classifications if needed
3. Define appropriate schemas for data transformation

### Adding New Intents
1. Update the intent classification prompt
2. Add corresponding API entries to the registry
3. Test with example inputs

## Development

### Project Structure
```
smart-router-azure/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── .env                # Environment configuration
├── example_input.json  # Sample request payload
├── venv/               # Virtual environment
└── README.md
```

### Testing
Use the provided example input file to test the service:
```bash
curl -X POST http://localhost:8000/smart-router \
  -H "Content-Type: application/json" \
  -d @example_input.json
```

## License

This project is for development and testing purposes.
