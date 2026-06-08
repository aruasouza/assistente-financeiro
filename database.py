# database.py - Gerenciamento de bancos de dados SQLite

import sqlite3
import os
import bcrypt
from pathlib import Path

DB_DIR = Path("./databases")
DB_DIR.mkdir(exist_ok=True)

USERS_DB = DB_DIR / "users.db"


def get_users_db():
    conn = sqlite3.connect(str(USERS_DB))
    conn.row_factory = sqlite3.Row
    return conn


def get_user_db(username: str):
    db_path = DB_DIR / f"{username}.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_users_db():
    conn = get_users_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def init_user_financial_db(username: str):
    conn = get_user_db(username)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL CHECK(type IN ('income', 'expense')),
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def register_user(username: str, password: str) -> tuple[bool, str]:
    init_users_db()
    try:
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        conn = get_users_db()
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.lower().strip(), password_hash),
        )
        conn.commit()
        conn.close()
        init_user_financial_db(username.lower().strip())
        return True, "Usuário cadastrado com sucesso!"
    except sqlite3.IntegrityError:
        return False, "Nome de usuário já existe. Escolha outro."
    except Exception as e:
        return False, f"Erro ao cadastrar: {str(e)}"


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    init_users_db()
    try:
        conn = get_users_db()
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username.lower().strip(),),
        ).fetchone()
        conn.close()
        if row is None:
            return False, "Usuário não encontrado."
        if bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            init_user_financial_db(username.lower().strip())
            return True, "Autenticado com sucesso!"
        return False, "Senha incorreta."
    except Exception as e:
        return False, f"Erro na autenticação: {str(e)}"


def add_transaction(username: str, type_: str, amount: float, category: str, description: str, date: str) -> tuple[bool, str]:
    try:
        conn = get_user_db(username)
        conn.execute(
            """INSERT INTO transactions (type, amount, category, description, date)
               VALUES (?, ?, ?, ?, ?)""",
            (type_, amount, category, description, date),
        )
        conn.commit()
        conn.close()
        return True, "Transação registrada com sucesso!"
    except Exception as e:
        return False, f"Erro ao registrar transação: {str(e)}"


def run_sql_query(username: str, sql: str) -> tuple[bool, list[dict] | str]:
    """Executa uma consulta SQL arbitrária (somente SELECT) no banco do usuário."""
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        return False, "Apenas consultas SELECT são permitidas."
    try:
        conn = get_user_db(username)
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return True, rows
    except Exception as e:
        return False, f"Erro na consulta SQL: {str(e)}"


def get_schema(username: str) -> str:
    """Retorna o schema do banco do usuário para o LLM usar."""
    return """
    Tabela: transactions
    Colunas:
      - id (INTEGER): identificador único
      - type (TEXT): 'income' para receita, 'expense' para gasto
      - amount (REAL): valor em reais
      - category (TEXT): categoria da transação
      - description (TEXT): descrição livre
      - date (TEXT): data no formato YYYY-MM-DD
      - created_at (TIMESTAMP): quando foi registrado
    """
