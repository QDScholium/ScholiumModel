from api.model import RAG

import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import HumanMessage


from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from api.model_utils import filter_results

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DB_URI = os.environ.get("DB_URI")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
model = ChatOpenAI(
    model="gpt-4o",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(api_key= pinecone_api_key)
index = pc.Index("scholium-index")
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

async def invoke_chat(query:str, thread: str):
    input_messages = [HumanMessage(query)]
    config = {"configurable": {"thread_id": thread}}
    output = await RAG.ainvoke({"messages": input_messages},config)
    output["messages"][-1].pretty_print()

async def test_chat():
    thread = "123"
    await invoke_chat("Hello", thread)
    await invoke_chat("dog",thread)
    
def test_filter_results():
    retrieved_docs = vector_store.similarity_search_with_score("Dogs are really cute", k=1) 
    filter_results(retrieved_docs)
    assert(len(retrieved_docs) == 0)

def draw_graph(graph):
    from IPython.display import Image, display
    image_obj = Image(graph.get_graph().draw_mermaid_png())
    with open("graph.png", "wb") as f:
        f.write(image_obj.data)


if __name__ == "__main__":

    import asyncio  
    # asyncio.run(test_chat())
    # test_filter_results()
    # draw_graph(RAG)