from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from categories import EXPENSE_CATEGORIES, INCOME_CATEGORIES

class Record(BaseModel):
    username: str = Field(description="Nome de usuário.")
    type: Literal['income','expense'] = Field(description="Tipo de transação.")
    amount: float = Field(description="Valor da transação.",gt=0.0)
    category: str = Field(description="Categoria da transação.")
    description: str = Field(description="Descrição da transação.")
    date: str = Field(description="Data da transação no formato YYYY-MM-DD.")

class UsuarioExiste(BaseModel):
    username: Optional[str] = Field(description="Nome de usuário, se existir.")
    new_user: bool = Field(description="True se este é um novo usuário que deseja realizar o cadastro.")

class Chart(BaseModel):
    type: Literal['bar', 'line', 'pie']  = Field(description="Tipo de gráfico a ser gerado.")
    x_axis: str = Field(description="Nome da coluna a ser usada no eixo X.")
    y_axis: str = Field(description="Nome da coluna a ser usada no eixo Y.")
    title: str = Field(description="Título do gráfico.")
    relevant: bool = Field(description="True se o gráfico for relevante para a pergunta do usuário. Em perguntas sobre transações específicas, sobre saldo ou pedidos de registro de transações o gráfico não é relevante. Em perguntas cuja resposta é um único valor o gráfico também não é relevante.")

class Resposta(BaseModel):
    content: str = Field(description="Resposta em texto para o usuário.")
    chart: Optional[Chart] = Field(description="Detalhes para gerar um gráfico, se necessário.")
    chart_data: Optional[str] = Field(description="Dados para o gráfico no formato csv.")