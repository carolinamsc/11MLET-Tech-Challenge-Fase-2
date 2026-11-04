"""Verifica a regra do desafio: funções com no máximo 20 linhas de corpo.

Conta linhas não vazias do corpo de cada função (excluindo docstring).
Uso: ``poetry run python scripts/check_function_length.py``
"""

import ast
from pathlib import Path

MAX_BODY_LINES = 20
TARGET_DIRS = ("src", "scripts")


def _body_line_count(node: ast.FunctionDef, source_lines: list[str]) -> int:
    """Conta linhas não vazias do corpo, ignorando a docstring."""
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if not body:
        return 0
    start, end = body[0].lineno, node.end_lineno or body[-1].lineno
    return sum(1 for line in source_lines[start - 1 : end] if line.strip())


def _check_file(path: Path) -> list[str]:
    """Retorna violações (função > 20 linhas) de um arquivo."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            count = _body_line_count(node, lines)
            if count > MAX_BODY_LINES:
                violations.append(f"{path}:{node.lineno} {node.name}() tem {count} linhas")
    return violations


def main() -> int:
    """Varre src/ e scripts/ e reporta funções acima do limite."""
    violations: list[str] = []
    for directory in TARGET_DIRS:
        for path in sorted(Path(directory).rglob("*.py")):
            violations.extend(_check_file(path))
    if violations:
        print("Funções acima de 20 linhas:")
        print("\n".join(f"  - {v}" for v in violations))
        return 1
    print(f"OK: todas as funções em {TARGET_DIRS} têm <= {MAX_BODY_LINES} linhas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
