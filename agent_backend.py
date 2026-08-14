from __future__ import annotations
import os
import operator
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# =========================================================
# LLM Setup with Fallback Protection (Groq -> Google Gemini)
# =========================================================
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=3
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Groq will execute by default; if Rate Limits hit, Gemini automatically takes over
llm = groq_llm.with_fallbacks([gemini_llm])


# =========================================================
# Pydantic Schemas
# =========================================================
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    brief: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (250–500 words for deep technical coverage).",
    )
    
    # FIX: Expanded allowed section types so LLM doesn't crash validation
    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion",
        "news_roundup", "system_design", "case_study", "benchmark"
    ] = Field(
        default="core",
        description="Section classification type.",
    )
    
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="Who this blog is for.")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp).")
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

    # Validator to fix String vs Boolean validation errors from smaller LLMs
    @field_validator("needs_research", mode="before")
    @classmethod
    def parse_boolean(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    plan: Optional[Plan]
    needs_research: bool
    mode: str
    queries: List[str]
    evidence: List[EvidenceItem]
    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str


# =========================================================
# Graph Nodes
# =========================================================
def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(
                content="""You are a routing module for a blog planner.

CRITICAL JSON FORMATTING RULE:
- 'needs_research' MUST be a raw JSON boolean: true or false (WITHOUT QUOTES). Do NOT output "true" or "false" as strings.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific.
"""
            ),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research") else "orchestrator"


def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        raw_snippet = r.get("content") or r.get("snippet") or ""
        
        # FIX: Truncate long snippets to prevent tool validation crash
        clean_snippet = raw_snippet[:250].strip() + "..." if len(raw_snippet) > 250 else raw_snippet.strip()

        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": clean_snippet,
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


def research_node(state: State) -> dict:
    queries = state.get("queries", []) or []
    max_results = 3
    raw_results: List[dict] = []

    # FIX: Top 2 queries evaluate karo to avoid token exhaustion
    for q in queries[:2]:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    # FIX: Send only essential clean fields to LLM
    clean_results = [
        {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
        for r in raw_results[:5]
    ]

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(
                content="""You are a research synthesizer for writing.
Given raw web search results, produce a deduplicated list of EvidenceItem objects.
Rules:
- Only include items with a non-empty url.
- Keep snippets short.
- Deduplicate by URL.
"""
            ),
            HumanMessage(content=f"Raw results:\n{clean_results}"),
        ]
    )
    
    dedup = {e.url: e for e in pack.evidence if e.url}
    return {"evidence": list(dedup.values())}


def orchestrator(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    # Shorten evidence snippets to avoid TPM rate-limits
    short_evidence = [
        {
            "title": e.title,
            "url": e.url,
            "snippet": (e.snippet[:150] + "...") if e.snippet else "",
        }
        for e in evidence[:5]
    ]

    plan = planner.invoke(
        [
            SystemMessage(
                content="""You are a Lead AI Research Analyst and Senior Developer Advocate.
Your goal is to produce a comprehensive, long-form, highly detailed technical blog outline.

Hard Requirements:
- Plan 4–6 comprehensive sections (tasks).
- Assign a generous target_words count between 250 and 500 for each task.
- Ensure the blog covers technical architecture, real-world trade-offs, developer/business impact, and benchmark comparisons where applicable.
- Output must strictly match the Plan schema.
"""
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims):\n"
                    f"{short_evidence}"
                )
            ),
        ]
    )
    return {"plan": plan}


def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])[:4]],
            },
        )
        for task in state["plan"].tasks
    ]


def worker(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- Title: {e.title} | URL: {e.url}"
            for e in evidence[:5]
        )

    section_md = llm.invoke(
        [
            SystemMessage(
                content="""You are a Lead AI Research Analyst writing a long-form, highly technical, and exhaustive article section in Markdown.

DEPTH & WRITING STYLE INSTRUCTIONS:
- Do NOT write superficial 1-2 sentence summaries. Deep-dive into each bullet point thoroughly.
- For each bullet point, elaborate on:
  1) Technical Mechanics / Architecture (How it actually works under the hood).
  2) Real-World Impact / Implications (Why developers or enterprise teams should care).
  3) Trade-offs, Performance Benchmarks, or Regulatory/Operational nuances.
- Maintain a professional, crisp, and authoritative technical tone (like a Senior Engineer or Tech Lead writing on Substack/Medium).
- Aim to fulfill the target word count (~250-500 words per section) with substantive technical insight, not fluff.

LINK CITATION INSTRUCTIONS (CRITICAL):
- Never use generic link anchor text like '[Read More]', '[Link]', '[Source]', or raw URLs.
- ALWAYS embed links into descriptive, context-rich anchor text within sentences.
  Example Bad: "Google released updates. [Read More](https://...)"
  Example Good: "According to the latest details on [Google's July 2026 AI Infrastructure Release](https://...), the new Gemini features..."
- Only cite URLs provided in the Evidence list.
"""
            ),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these for links):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


def reducer(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [
        md
        for _, md in sorted(
            state["sections"],
            key=lambda x: x[0]
        )
    ]

    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    return {"final": final_md}


# Workflow graph setup
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router", route_next, {"research": "research", "orchestrator": "orchestrator"}
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

if __name__ == "__main__":
    out = app.invoke({"topic": "Write a blog on Self Attention", "sections": []})
    print(out["final"])