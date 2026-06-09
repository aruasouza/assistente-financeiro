# agent.py - Agente financeiro com LangGraph + Gemini

import json
import os
import re
from datetime import date
from typing import Annotated, Any, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from categories import ALL_CATEGORIES, EXPENSE_CATEGORIES, INCOME_CATEGORIES
from database import (
    authenticate_user,
    get_schema,
    register_user,
)
from tools import build_record_tool, build_sql_tool
from outputs import Resposta, UsuarioExiste


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
            "messages": [AIMessage(content='👋 Olá! Para acessar seu assistente financeiro, informe seu **nome de usuário** (ou diga: "Não tenho cadastro" para criar um):')],
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
        if (lower == 'não tenho cadastro') or (lower == 'nao tenho cadastro'):
            return {
                "register_pending": True,
                "messages": [AIMessage(content="Vamos criar seu cadastro!\nQual será o seu nome de usuário?")]
            }            
        elif re.match(r'^[a-z0-9_]{3,20}$', last_user_msg):
            return {
                "messages": [AIMessage(content="Agora digite sua senha:")],
                "username": last_user_msg,
            }
        return {
                "messages": [AIMessage(content="Por favor, informe um nome de usuário válido para continuar. Utilize apenas letras minúsculas, números e '_'.")],
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
    """Nó principal para usuários autenticados.

    Fluxo de duas chamadas ao LLM:
      1ª — tool_calling: o modelo decide quais ferramentas acionar (registro ou SQL).
      2ª — structured_output: com os resultados das tools anexados, gera a Resposta final.
    Isso garante exatamente dois round-trips ao LLM, minimizando latência.
    """
    username = state["username"]
    schema = get_schema(username)
    system_prompt = FINANCIAL_SYSTEM.format(
        schema=schema,
        expense_cats=", ".join(EXPENSE_CATEGORIES),
        income_cats=", ".join(INCOME_CATEGORIES),
        today=date.today().isoformat(),
    )

    # Ferramentas disponíveis na 1ª chamada (registro e consulta SQL)
    record_tool = build_record_tool(username)
    sql_tool = build_sql_tool(username)
    action_tools = {t.name: t for t in [record_tool, sql_tool]}

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)

    # ── 1ª chamada: tool_calling ──────────────────────────────────────────────
    llm_with_tools = llm.bind_tools(list(action_tools.values()))
    messages_1a_chamada = [SystemMessage(content=system_prompt)] + state["messages"]
    ai_msg = llm_with_tools.invoke(messages_1a_chamada)

    # ── Execução das tools ────────────────────────────────────────────────────
    tool_results: list[ToolMessage] = []
    for call in ai_msg.tool_calls or []:
        tool = action_tools.get(call["name"])
        if tool is None:
            result = f"Ferramenta '{call['name']}' não encontrada."
        else:
            try:
                result = tool.invoke(call["args"])
            except Exception as e:
                result = f"Erro ao executar '{call['name']}': {e}"
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )

    # ── 2ª chamada: structured_output ────────────────────────────────────────
    llm_structured = llm.with_structured_output(Resposta)
    messages_2a_chamada = messages_1a_chamada + [ai_msg] + tool_results
    resposta: Resposta = llm_structured.invoke(messages_2a_chamada)

    # ── Monta retorno do nó ───────────────────────────────────────────────────
    chart = resposta.chart
    if chart is not None and chart.relevant:
        return {
            "messages": [AIMessage(content=resposta.content)],
            "chart_request": {"config": chart, "data": resposta.chart_data},
        }
    return {"messages": [AIMessage(content=resposta.content)], "chart_request": None}


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