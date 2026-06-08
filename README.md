# 💼 FinAgent — Assistente Financeiro com IA

Agente financeiro pessoal construído com **LangGraph + Gemini + Streamlit**.

## Arquitetura

```
financial_agent/
├── app.py           # Interface Streamlit
├── agent.py         # Grafo LangGraph + lógica do agente
├── database.py      # Gerenciamento SQLite (usuários + finanças por usuário)
├── categories.py    # Categorias pré-definidas de gastos e receitas
├── requirements.txt
├── .env             # Sua API Key (não commitar!)
└── databases/       # Criado automaticamente
    ├── users.db     # Banco de usuários (autenticação)
    └── {username}.db # Um banco por usuário (isolamento total)
```

## Fluxo do Agente (LangGraph)

```
START
  │
  ├── [não autenticado] → auth_node
  │     ├── idle → pede username
  │     ├── awaiting_username → pede senha (ou rota de cadastro)
  │     ├── awaiting_password → autentica
  │     ├── awaiting_register_username → pede username novo
  │     └── awaiting_register_password → cria conta
  │
  └── [autenticado] → financial_node
        ├── registrar transação (income/expense)
        ├── consulta SQL arbitrária
        └── gerar gráfico (bar/line/pie)
END
```

## Setup

### 1. Instalar dependências

```bash
cd financial_agent
pip install -r requirements.txt
```

### 2. Configurar API Key

Crie um arquivo `.env`:
```env
GEMINI_API_KEY=sua_chave_aqui
```

Ou insira diretamente na barra lateral do app.

### 3. Executar

```bash
streamlit run app.py
```

## Funcionalidades

### Autenticação
- Login com usuário e senha (senha hasheada com bcrypt)
- Cadastro direto pelo chat
- Isolamento completo por usuário (banco SQLite separado)

### Registro de Transações
O agente extrai automaticamente da linguagem natural:
- **Tipo**: gasto ou receita
- **Valor**: em reais
- **Categoria**: classificada automaticamente pelo LLM
- **Descrição**: o que foi o gasto/receita
- **Data**: a data mencionada ou hoje

Exemplos:
> "Gastei 85 reais no mercado hoje"
> "Recebi meu salário de 5000 reais ontem"
> "Paguei 120 de streaming esse mês"

### Consultas e Gráficos
O agente gera SQL arbitrário para responder qualquer pergunta:
> "Quanto gastei com alimentação esse mês?"
> "Mostre meus gastos por categoria em um gráfico de pizza"
> "Qual meu saldo dos últimos 3 meses?"
> "Quais foram meus 5 maiores gastos?"

### Categorias de Gasto
Alimentação, Moradia, Transporte, Saúde, Educação, Lazer e Entretenimento,
Vestuário, Tecnologia, Serviços e Assinaturas, Impostos e Taxas, Pets,
Viagem, Presentes e Doações, Emergência, Outros

### Categorias de Receita
Salário, Freelance, Aluguel Recebido, Dividendos e Investimentos,
Venda de Bens, Bonificação, Pensão / Benefícios, Reembolso, Renda Extra, Outros

## Atalhos na Sidebar (após login)
- 📊 Resumo do mês
- 📈 Maiores gastos com gráfico
- 💰 Saldo atual
- 📅 Últimas transações
