import os
import google.generativeai as genai
import uuid
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langgraph.graph import StateGraph
from langsmith import trace
import json
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Cache Setup ---
CACHE_DIR = "./gemini_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

import hashlib

def get_cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()

def save_to_cache(prompt: str, response: str):
    key = get_cache_key(prompt)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump({"prompt": prompt, "response": response}, f)

def load_from_cache(prompt: str) -> Optional[str]:
    key = get_cache_key(prompt)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data["response"]
    return None

# --- Gemini Flash with caching ---
def gemini_flash(prompt: str) -> str:
    cached = load_from_cache(prompt)
    if cached:
        print("⚡ Using cached Gemini response")
        return cached

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        text = response.text
        save_to_cache(prompt, text)
        return text
    except Exception as e:
        print(f"⚠ Gemini API error: {e}")
        if cached:
            return cached
        return "Gemini API unavailable and no cached result."

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

# --- Persistent User History ---
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

# --- LangChain Tools ---
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
    summaries = [b.synthesis for b in history[-3:]]
    prompt = f"Summarize the following prior research briefs for context:\n" + "\n---\n".join(summaries)
    output = gemini_flash(prompt)
    state.context_summary = output
    return state

def planning_node(state: GraphState) -> GraphState:
    topic = state.topic
    depth = state.depth
    context_summary = state.context_summary
    parser = PydanticOutputParser(pydantic_object=ResearchPlanStep)
    prompt = (f"Plan research steps for topic '{topic}' at depth {depth}."
              " For each step, respond ONLY with a single valid JSON object (not a list) matching this schema:\n"
              f"{parser.get_format_instructions()}")
    if context_summary:
        prompt += f"\nContext: {context_summary}"
    steps = []
    for i in range(depth):
        step_prompt = prompt + f"\nStep {i+1}:"
        for _ in range(3):
            try:
                output = gemini_flash(step_prompt)
                try:
                    output_json = json.loads(output)
                    if isinstance(output_json, list):
                        for item in output_json:
                            if isinstance(item, dict):
                                output = json.dumps(item)
                                break
                        else:
                            raise ValueError("No valid dict in Gemini output list")
                    elif isinstance(output_json, dict):
                        output = json.dumps(output_json)
                    else:
                        raise ValueError("Gemini output is not a dict or list")
                except Exception:
                    pass
                step = parser.parse(output)
                step.step_id = str(uuid.uuid4())
                steps.append(step)
                break
            except ValidationError:
                continue
    state.steps = steps
    return state

def search_node(state: GraphState) -> GraphState:
    steps = state.steps
    sources = []
    for step in steps:
        query = step.description
        results = search_tool.run(query)
        sources.extend(results[:2])
    state.source_urls = [s["url"] for s in sources if isinstance(s, dict) and "url" in s]
    state.source_titles = [s["title"] for s in sources if isinstance(s, dict) and "title" in s]
    return state

def content_fetch_node(state: GraphState) -> GraphState:
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
graph = StateGraph(GraphState)
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
graph.add_edge("__start__", "context_summarization")
compiled_graph = graph.compile()

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
