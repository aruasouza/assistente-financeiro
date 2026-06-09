from pydantic import BaseModel
from database import add_transaction, run_sql_query
from outputs import Resposta
from typing import Type
from langchain_core.tools import tool

def build_record_tool(username):
    @tool
    def criar_registro(type: str, amount: float, category: str, description: str, date: str) -> str:
        """Cria um novo registro de transação financeira (receita ou despesa)."""
        result = add_transaction(username, type, amount, category, description, date)
        return result[1]
        
    return criar_registro

def build_sql_tool(username):
    @tool
    def sqltool(query: str) -> str:
        """Use essa ferramenta para consultar a base sqlite e recuperar dados financeiros do usuário."""
        success, result = run_sql_query(username, query)
        return result
    return sqltool

@tool
def responder_usuario(resposta: Resposta):
    """Esta ferramenta deve ser usada para responder à solicitação do usuário."""
    return 'Ok'