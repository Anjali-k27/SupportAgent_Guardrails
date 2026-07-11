"""
Enterprise AI Support Platform
Session 6 of 12 — Guardrails & Execution Bounding

Extends Session 5 with security sandwich:
ingress_node (PII + injection detection),
egress_node (output validation),
blocked_response_node (zero-token refusal).

Run server: python api.py  → http://localhost:8000
Run CLI:    python support_agent.py
"""

import os
import re
import time
import operator
import json
import uuid
import sqlite3
from typing import TypedDict, Annotated, Literal, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
