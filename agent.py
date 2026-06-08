# agent.py - Agente financeiro com LangGraph + Gemini

import json
import re
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from categories import ALL_CATEGORIES, EXPENSE_CATEGORIES, INCOME_CATEGORIES
from database import (
    add_transaction,
    authenticate_user,
    get_schema,
    register_user,
    run_sql_query,
)


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    authenticated: bool
    username: str | None
    auth_step: str          # "idle" | "awaiting_username" | "awaiting_password" | "awaiting_register_username" | "awaiting_register_password"
    pending_username: str | None
    chart_request: dict | None   # dados para gerar gráfico no Streamlit


# ─── LLM ──────────────────────────────────────────────────────────────────────

def make_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.3,
    )


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

Quando o usuário quiser registrar uma transação, responda com JSON no formato:
```json
{{"action": "add_transaction", "type": "expense"|"income", "amount": 123.45, "category": "Categoria", "description": "descrição", "date": "YYYY-MM-DD"}}
```

Quando precisar consultar dados, responda com JSON no formato:
```json
{{"action": "sql_query", "sql": "SELECT ...", "chart": null}}
```
Ou com gráfico:
```json
{{"action": "sql_query", "sql": "SELECT ...", "chart": {{"type": "bar"|"line"|"pie", "x": "coluna_x", "y": "coluna_y", "title": "Título"}}}}
```

Se a pergunta é conversacional e não precisa de ação, responda normalmente em texto."""


# ─── Helper ───────────────────────────────────────────────────────────────────

def extract_json_action(text: str) -> dict | None:
    """Extrai bloco JSON de ação da resposta do LLM."""
    text = text[0]['text']
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Tenta JSON inline
    match2 = re.search(r'(\{"action".*?\})', text, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(1))
        except Exception:
            pass
    return None


def clean_response(text: str) -> str:
    """Remove blocos JSON da resposta visível ao usuário."""
    text = text[0]['text']
    cleaned = re.sub(r"```json\s*\{.*?\}\s*```", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ─── Nodes ────────────────────────────────────────────────────────────────────

def auth_node(state: AgentState, llm) -> dict:
    """Gerencia o fluxo de autenticação e cadastro."""
    messages = state["messages"]
    auth_step = state.get("auth_step", "idle")
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.strip()
            break

    lower = last_user_msg.lower()

    # ── Cadastro ──────────────────────────────────────
    if auth_step == "awaiting_register_username":
        return {
            "auth_step": "awaiting_register_password",
            "pending_username": last_user_msg,
            "messages": [AIMessage(content=f"Ótimo! Agora escolha uma senha para o usuário **{last_user_msg}**:")],
            "chart_request": None,
        }

    if auth_step == "awaiting_register_password":
        username = state.get("pending_username", "")
        ok, msg = register_user(username, last_user_msg)
        if ok:
            return {
                "auth_step": "idle",
                "pending_username": None,
                "authenticated": True,
                "username": username.lower().strip(),
                "messages": [AIMessage(content=f"✅ Cadastro realizado com sucesso! Bem-vindo(a), **{username}**! Como posso ajudar nas suas finanças?")],
                "chart_request": None,
            }
        else:
            return {
                "auth_step": "awaiting_register_username",
                "pending_username": None,
                "messages": [AIMessage(content=f"❌ {msg} Escolha outro nome de usuário:")],
                "chart_request": None,
            }

    # ── Login ─────────────────────────────────────────
    if auth_step == "awaiting_username":
        # Verificar se quer se cadastrar
        if any(w in lower for w in ["não tenho", "nao tenho", "cadastrar", "criar conta", "não possuo", "nao possuo"]):
            return {
                "auth_step": "awaiting_register_username",
                "messages": [AIMessage(content="Vamos criar sua conta! Escolha um nome de usuário:")],
                "chart_request": None,
            }
        return {
            "auth_step": "awaiting_password",
            "pending_username": last_user_msg,
            "messages": [AIMessage(content=f"Olá, **{last_user_msg}**! Agora informe sua senha:")],
            "chart_request": None,
        }

    if auth_step == "awaiting_password":
        username = state.get("pending_username", "")
        ok, msg = authenticate_user(username, last_user_msg)
        if ok:
            return {
                "auth_step": "idle",
                "pending_username": None,
                "authenticated": True,
                "username": username.lower().strip(),
                "messages": [AIMessage(content=f"✅ Autenticado! Bem-vindo(a) de volta, **{username}**! Como posso ajudar nas suas finanças hoje?")],
                "chart_request": None,
            }
        else:
            return {
                "auth_step": "awaiting_username",
                "pending_username": None,
                "messages": [AIMessage(content=f"❌ {msg} Tente novamente — informe seu nome de usuário (ou diga que não tem cadastro):")],
                "chart_request": None,
            }

    # ── Estado idle: iniciar autenticação ─────────────
    if any(w in lower for w in ["cadastrar", "criar conta", "nova conta", "quero me cadastrar"]):
        return {
            "auth_step": "awaiting_register_username",
            "messages": [AIMessage(content="Vamos criar sua conta! Escolha um nome de usuário:")],
            "chart_request": None,
        }

    # Primeiro contato ou mensagem genérica → pedir login
    return {
        "auth_step": "awaiting_username",
        "messages": [AIMessage(content="👋 Olá! Para acessar seu assistente financeiro, informe seu **nome de usuário** (ou diga que não tem cadastro para criar um):")],
        "chart_request": None,
    }


def financial_node(state: AgentState, llm) -> dict:
    """Nó principal para usuários autenticados."""
    username = state["username"]
    schema = get_schema(username)
    system_prompt = FINANCIAL_SYSTEM.format(
        schema=schema,
        expense_cats=", ".join(EXPENSE_CATEGORIES),
        income_cats=", ".join(INCOME_CATEGORIES),
        today=date.today().isoformat(),
    )

    # Monta histórico para o LLM
    history = [SystemMessage(content=system_prompt)]
    for m in state["messages"]:
        history.append(m)

    response = llm.invoke(history)
    response_text = response.content

    # Verifica se há ação JSON
    action = extract_json_action(response_text)
    visible_text = clean_response(response_text)
    chart_request = None

    if action:
        act = action.get("action")

        if act == "add_transaction":
            ok, msg = add_transaction(
                username=username,
                type_=action.get("type", "expense"),
                amount=float(action.get("amount", 0)),
                category=action.get("category", "Outros"),
                description=action.get("description", ""),
                date=action.get("date", date.today().isoformat()),
            )
            type_label = "💰 Receita" if action.get("type") == "income" else "💸 Gasto"
            emoji = "✅" if ok else "❌"
            reply = (
                f"{emoji} {type_label} de **R$ {action.get('amount', 0):.2f}** "
                f"— {action.get('description', '')} "
                f"[{action.get('category')}] em {action.get('date')} registrado!"
                if ok
                else f"❌ Erro: {msg}"
            )
            if visible_text:
                reply = visible_text + "\n\n" + reply
            return {"messages": [AIMessage(content=reply)], "chart_request": None}

        elif act == "sql_query":
            sql = action.get("sql", "")
            ok, result = run_sql_query(username, sql)
            if ok:
                chart_cfg = action.get("chart")
                if chart_cfg and isinstance(result, list) and result:
                    chart_request = {"config": chart_cfg, "data": result}
                    reply = visible_text or "Aqui está o gráfico com os dados solicitados:"
                else:
                    if isinstance(result, list) and result:
                        # Formata tabela simples
                        headers = list(result[0].keys())
                        rows_text = "\n".join(
                            " | ".join(str(row.get(h, "")) for h in headers)
                            for row in result[:50]
                        )
                        reply = (visible_text or "Resultado da consulta:") + f"\n\n```\n{' | '.join(headers)}\n{rows_text}\n```"
                    else:
                        reply = visible_text or "Nenhum dado encontrado para a consulta."
            else:
                reply = f"❌ Erro na consulta: {result}"
            return {"messages": [AIMessage(content=reply)], "chart_request": chart_request}

    return {"messages": [AIMessage(content=response_text)], "chart_request": None}


# ─── Router ───────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    if state.get("authenticated"):
        return "financial"
    return "auth"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph(api_key: str):
    llm = make_llm(api_key)

    def _auth(state):
        return auth_node(state, llm)

    def _financial(state):
        return financial_node(state, llm)

    builder = StateGraph(AgentState)
    builder.add_node("auth", _auth)
    builder.add_node("financial", _financial)

    builder.add_conditional_edges(START, route, {"auth": "auth", "financial": "financial"})
    builder.add_edge("auth", END)
    builder.add_edge("financial", END)

    return builder.compile()


def get_initial_state() -> AgentState:
    return AgentState(
        messages=[],
        authenticated=False,
        username=None,
        auth_step="idle",
        pending_username=None,
        chart_request=None,
    )
