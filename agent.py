# agent.py - Agente financeiro com LangGraph + Gemini

import json
import os
import re
from datetime import date
from typing import Annotated, Any, TypedDict
from outputs import UsuarioExiste

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from categories import ALL_CATEGORIES, EXPENSE_CATEGORIES, INCOME_CATEGORIES
from database import (
    authenticate_user,
    get_schema,
    register_user,
)
from tools import build_record_tool, build_sql_tool, responder_usuario


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    authenticated: bool
    username: str
    first_interaction: bool
    register_pending: bool
    chart_request: dict | None


# ─── System Prompts ───────────────────────────────────────────────────────────

AUTH_SYSTEM = """Você é um assistente financeiro pessoal amigável e profissional.
No momento o usuário NÃO está autenticado. Sua única função agora é ajudar no processo de autenticação ou cadastro.
Seja educado e sucinto. Não discuta finanças até o usuário estar autenticado."""

FINANCIAL_SYSTEM = """Você é um assistente financeiro pessoal inteligente e empático.
O usuário está autenticado. Você pode:
1. Registrar entradas (receitas) e saídas (gastos) — extraia tipo, valor, descrição e data da mensagem.
2. Responder perguntas sobre as finanças gerando consultas SQL.
3. Sugerir gráficos quando relevante.

Schema do banco de dados do usuário:
{schema}

Categorias de GASTO (expense): {expense_cats}
Categorias de RECEITA (income): {income_cats}

Hoje é {today}.

Toda resposta ao usuário deve ser entregue através da ferramenta 'responder_usuario'.
Se o usuário pedir para registrar receitas ou despesas não retorne nenhum dado ou gráfico."""

# ─── Nodes ────────────────────────────────────────────────────────────────────

def auth_node(state: AgentState) -> dict:
    """Gerencia o fluxo de autenticação e cadastro."""
    if state['first_interaction']:
        return {
            "messages": [AIMessage(content="👋 Olá! Para acessar seu assistente financeiro, informe seu **nome de usuário** (ou diga que não tem cadastro para criar um):")],
            'first_interaction': False
        }
    messages = state["messages"]
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.strip()
            break

    lower = last_user_msg.lower()
    if state.get('username') is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.0,
        )
        llm_estruturado = llm.with_structured_output(UsuarioExiste)
        resp = llm_estruturado.invoke(state['messages'])
        if resp.new_user:
            return {
                "register_pending": True,
                "messages": [AIMessage(content="Vamos criar seu cadastro!\nQual será o seu nome de usuário?")]
            }            
        if 'username' in resp.model_fields_set:
            return {
                "messages": [AIMessage(content="Agora digite sua senha:")],
                "username": resp.username,
            }
        return {
                "messages": [AIMessage(content="Por favor, informe seu nome de usuário para continuar.")],
            }
    if state.get('username'):
        username = state['username']
        password = last_user_msg
        ok, msg = authenticate_user(username, password)
        if ok:
            return {
                "authenticated": True,
                "messages": [AIMessage(content=f"✅ Autenticado! Bem-vindo(a) de volta, {username}! Como posso ajudar nas suas finanças hoje?")],
            }
        else:
            return {
                "messages": [AIMessage(content=f"❌ {msg}\n\nCredenciais inválidas. Informe novamente seu nome de usuário:")],
                "username": None,
            }
    

def register_node(state: AgentState) -> dict:
    """Registra um novo usuário."""
    if state['first_interaction']:
        return {
            "first_interaction": False
        }
    messages = state["messages"]
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.strip()
            break
    if not state.get("username"):
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', last_user_msg):
            return {
                "messages": [AIMessage(content="O nome de usuário deve ter entre 3 e 20 caracteres e conter apenas letras, números e underscores (_)")],
            }
        return {
            "messages": [AIMessage(content="Qual será a sua senha?")],
            "username": last_user_msg,
        }
    else:
        username = state["username"]
        password = last_user_msg
        if len(password) < 4:
            return {
                "messages": [AIMessage(content="A senha deve ter pelo menos 4 caracteres.")],
            }
        ok, msg = register_user(username, password)
        if ok:
            return {
                "authenticated": True,
                "username": username,
                "messages": [AIMessage(content=msg)],
            }
        else:
            return {
                "messages": [AIMessage(content=f"❌ {msg}\n\nTente outro nome de usuário:")],
                "username": None,
            }


def financial_node(state: AgentState) -> dict:
    """Nó principal para usuários autenticados."""
    username = state["username"]
    schema = get_schema(username)
    system_prompt = FINANCIAL_SYSTEM.format(
        schema=schema,
        expense_cats=", ".join(EXPENSE_CATEGORIES),
        income_cats=", ".join(INCOME_CATEGORIES),
        today=date.today().isoformat(),
    )
    agent = create_react_agent(
        name="assistente-financeiro",
        model = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.3,
        ),
        tools=[build_record_tool(username),build_sql_tool(username),responder_usuario],
    )
    input = state.copy()
    input["messages"].insert(0, SystemMessage(content=system_prompt))
    response = agent.invoke(input)
    for msg in reversed(response["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "responder_usuario":
                    resposta_estruturada = tool_call["args"]["resposta"]
                    break
    chart = resposta_estruturada.get('chart')
    data = resposta_estruturada.get('chart_data')
    response_text = resposta_estruturada.get('content')
    if chart is not None:
        return {
            "messages": [AIMessage(content=response_text)], "chart_request": {
                "config": chart,
                "data": data
            }
        }
    else:
        return {"messages": [AIMessage(content=response_text)], "chart_request": None}


# ─── Router ───────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    if state.get("authenticated"):
        return "financial"
    if state.get("register_pending"):
        return "register"
    return "auth"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph():

    builder = StateGraph(AgentState)
    builder.add_node("auth", auth_node)
    builder.add_node("financial", financial_node)
    builder.add_node("register", register_node)
    builder.add_conditional_edges(START, route, {"auth": "auth", "financial": "financial","register": "register"})
    builder.add_edge("auth", END)
    builder.add_edge("financial", END)
    builder.add_edge("register", END)

    return builder.compile()


def get_initial_state() -> AgentState:
    return AgentState(
        messages=[],
        authenticated=False,
        username=None,
        chart_request=None,
        first_interaction = True,
        register_pending = False
    )
