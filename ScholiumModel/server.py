"""
This is for the copilotkit remote endpoint
"""
import os
from dotenv import load_dotenv 
load_dotenv()
from fastapi import FastAPI
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, Action as CopilotAction
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from ScholiumModel.model import RAG

app = FastAPI()

# Initialize the CopilotKit SDK
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="research_agent",
            description="Research agent.",
            graph=RAG,
        ),
    ],
)
 
# Add the CopilotKit endpoint to your FastAPI app
add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == '__main__':
    main()