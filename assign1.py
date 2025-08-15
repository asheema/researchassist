import os
import google.generativeai as genai
import uuid
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
#from langchain.tools import DuckDuckGoSearch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langgraph.graph import StateGraph
from langsmith import trace
import json
import json
import argparse
import sys

import os
from langchain_community.utilities import SerpAPIWrapper
import os
from dotenv import load_dotenv
load_dotenv()

#SERPAPI_API = os.getenv("SERPAPI_KEY")
#search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API)

import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
# ...other imports...

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def gemini_flash(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# --- Schemas ---


class ResearchPlanStep(BaseModel):
    step_id: str
    description: str
    rationale: str

class SourceSummary(BaseModel):
    source_url: str
    title: str
    summary: str
    evidence: List[str]

class FinalBrief(BaseModel):
    topic: str
    depth: int
    steps: List[ResearchPlanStep]
    sources: List[SourceSummary]
    synthesis: str
    references: List[str]
    context_summary: Optional[str] = None

class GraphState(BaseModel):
    topic: str
    depth: int
    follow_up: bool
    user_id: str
    context_summary: Optional[str] = None
    steps: Any = None
    source_urls: Any = None
    source_titles: Any = None
    documents: Any = None
    source_summaries: Any = None
    synthesis: str = None
    final_brief: Any = None    

# --- Persistent User History (simple file-based for demo) ---

USER_HISTORY_DIR = "./user_history"
os.makedirs(USER_HISTORY_DIR, exist_ok=True)

def get_user_history(user_id: str) -> List[FinalBrief]:
    path = os.path.join(USER_HISTORY_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return [FinalBrief(**item) for item in data]

def save_user_brief(user_id: str, brief: FinalBrief):
    history = get_user_history(user_id)
    history.append(brief)
    path = os.path.join(USER_HISTORY_DIR, f"{user_id}.json")
    with open(path, "w") as f:
        json.dump([b.dict() for b in history], f)

# --- LangChain LLMs and Tools ---

search_tool = DuckDuckGoSearchRun()

# --- Node Implementations ---

def context_summarization_node(state: GraphState) -> GraphState:
    user_id = state.user_id
    follow_up = state.follow_up
    if not follow_up:
        state.context_summary = None
        return state
    history = get_user_history(user_id)
    if not history:
        state.context_summary = None
        return state
    # Summarize prior briefs using LLM
    summaries = [b.synthesis for b in history[-3:]]
    prompt = f"Summarize the following prior research briefs for context:\n" + "\n---\n".join(summaries)
    output = gemini_flash(prompt)
    state.context_summary = output
    return state

def planning_node(state: GraphState) -> GraphState:
    topic = state.topic
    depth = state.depth
    context_summary = state.context_summary
    parser = PydanticOutputParser(pydantic_object=ResearchPlanStep)  # <-- Move this up!
    prompt = (f"Plan research steps for topic '{topic}' at depth {depth}."
          " For each step, respond ONLY with a single valid JSON object (not a list) matching this schema:\n"
          f"{parser.get_format_instructions()}"
)
    if context_summary:
        prompt += f"\nContext: {context_summary}"
    steps = []
    for i in range(depth):
     step_prompt = prompt + f"\nStep {i+1}:"
    for _ in range(3):  # Retry logic
        try:
            output = gemini_flash(step_prompt)
            # Try to parse as JSON, handle both object and list
            try:
                output_json = json.loads(output)
                # If it's a list, take the first dict-like element
                if isinstance(output_json, list):
                    # Find the first dict in the list
                    for item in output_json:
                        if isinstance(item, dict):
                            output = json.dumps(item)
                            break
                    else:
                        # If no dict found, raise error
                        raise ValueError("No valid dict in Gemini output list")
                elif isinstance(output_json, dict):
                    output = json.dumps(output_json)
                else:
                    raise ValueError("Gemini output is not a dict or list")
            except Exception:
                pass  # If not JSON, let parser handle it (may still work)
            step = parser.parse(output)
            step.step_id = str(uuid.uuid4())
            steps.append(step)
            break
        except ValidationError as e:
            # Optionally print for debugging:
            # print("Validation error:", e)
            continue
    state.steps = steps
    return state

def search_node(state: GraphState) -> GraphState:
    steps = state.steps
    sources = []
    for step in steps:
        query = step.description
        results = search_tool.run(query)
        print("Search results for query:", query)
        print(results)
        # Assume results is a list of dicts with 'url' and 'title'
        sources.extend(results[:2])  # Take top 2 per step
    state.source_urls = [s["url"] for s in sources if isinstance(s, dict) and "url" in s]
    state.source_titles = [s["title"] for s in sources if isinstance(s, dict) and "title" in s]
    return state

def content_fetch_node(state: GraphState) -> GraphState:
    # For demo, just pass URLs and titles forward
    state.documents = [
        Document(page_content=f"Content of {title}", metadata={"url": url})
        for url, title in zip(state.source_urls, state.source_titles)
    ]
    return state

def per_source_summarization_node(state: GraphState) -> GraphState:
    docs = state.documents
    summaries = []
    parser = PydanticOutputParser(pydantic_object=SourceSummary)
    for doc in docs:
        prompt = (f"Summarize the following source for research:\nTitle: {doc.metadata['url']}\nContent: {doc.page_content}\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
    f"{parser.get_format_instructions()}")
        for _ in range(3):
            try:
                output = gemini_flash(prompt)
                # Try to parse as JSON, handle both object and list
                try:
                    output_json = json.loads(output)
                    if isinstance(output_json, list):
                        output = json.dumps(output_json[0])
                except Exception:
                    pass
                summary = parser.parse(output)
                summary.source_url = doc.metadata["url"]
                summary.title = doc.page_content[:50]
                summaries.append(summary)
                break
            except ValidationError:
                continue
    state.source_summaries = summaries
    return state

def synthesis_node(state: GraphState) -> GraphState:
    topic = state.topic
    steps = state.steps
    summaries = state.source_summaries
    context_summary = state.context_summary
    prompt = f"Write a research brief for topic '{topic}' using the following steps and source summaries."
    if context_summary:
        prompt += f"\nContext: {context_summary}"
    prompt += "\nSteps:\n" + "\n".join([s.description for s in steps])
    prompt += "\nSources:\n" + "\n".join([s.summary for s in summaries])
    output = gemini_flash(prompt)
    state.synthesis = output
    return state

def post_processing_node(state: GraphState) -> GraphState:
    brief = FinalBrief(
        topic=state.topic,
        depth=state.depth,
        steps=state.steps,
        sources=state.source_summaries,
        synthesis=state.synthesis,
        references=[s.source_url for s in state.source_summaries],
        context_summary=state.context_summary,
    )
    state.final_brief = brief
    return state

# --- Graph Definition ---

# ...existing code...
from langgraph.graph import StateGraph

# --- Graph Definition ---
from pydantic import BaseModel
from typing import Dict, Any



# --- Graph Definition ---

graph = StateGraph(GraphState)
# ...rest of your graph construction code...

#graph = StateGraph()
graph.add_node("context_summarization", context_summarization_node)
graph.add_node("planning", planning_node)
graph.add_node("search", search_node)
graph.add_node("content_fetch", content_fetch_node)
graph.add_node("per_source_summarization", per_source_summarization_node)
graph.add_node("synthesis_step", synthesis_node)
graph.add_node("post_processing", post_processing_node)
graph.add_edge("context_summarization", "planning")
graph.add_edge("planning", "search")
graph.add_edge("search", "content_fetch")
graph.add_edge("content_fetch", "per_source_summarization")
graph.add_edge("per_source_summarization", "synthesis_step")
graph.add_edge("synthesis_step", "post_processing")
# Add a special start node and connect it to your entry node
graph.add_edge("__start__", "context_summarization")
compiled_graph = graph.compile()
# ...existing code...

# --- FastAPI Interface ---

app = FastAPI()

@app.post("/brief", response_model=FinalBrief)
def generate_brief(request: Dict[str, Any]):
    start_time = time.time()
    try:
        topic = request["topic"]
        depth = int(request["depth"])
        follow_up = bool(request["follow_up"])
        user_id = request["user_id"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body.")
    state = GraphState(
        topic=topic,
        depth=depth,
        follow_up=follow_up,
        user_id=user_id,
        context_summary=None,
    )
    with trace("research_brief", metadata={"user_id": user_id}):
        graph_output = compiled_graph.invoke(state)
    brief = graph_output["final_brief"]
    save_user_brief(user_id, brief)
    latency = time.time() - start_time
    # Token usage estimation (mocked)
    brief_dict = brief.dict()
    brief_dict["latency_seconds"] = latency
    brief_dict["token_estimate"] = len(str(brief_dict)) // 4
    return brief

# --- CLI Interface ---

def cli():
    parser = argparse.ArgumentParser(description="Research Brief Generator CLI")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--follow-up", action="store_true")
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()
    request = {
        "topic": args.topic,
        "depth": args.depth,
        "follow_up": args.follow_up,
        "user_id": args.user_id,
    }
    brief = generate_brief(request)
    print(brief.json(indent=2))

if __name__ == "__main__":
    if "cli" in sys.argv:
        cli()

