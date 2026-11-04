"""Valida o ambiente do desafio de intenção de compra.

Uso: ``poetry run python scripts/validate_env.py``
"""

import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = ("sklearn", "mlflow", "dvc", "pandas", "yaml")
REQUIRED_DIRS = ("src/purchase_intent", "configs", "tests")
MIN_PYTHON = (3, 11)


def check_python() -> bool:
    """Confere a versão mínima do Python."""
    ok = sys.version_info >= MIN_PYTHON
    print(f"[{'OK' if ok else 'FALHA'}] Python {sys.version.split()[0]} (mínimo 3.11)")
    return ok


def check_packages() -> bool:
    """Confere se todos os pacotes obrigatórios são importáveis."""
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "?")
            print(f"[OK] {package} {version}")
        except ImportError:
            print(f"[FALHA] pacote ausente: {package}")
            all_ok = False
    return all_ok


def check_structure() -> bool:
    """Confere a estrutura de diretórios e o arquivo .env."""
    all_ok = True
    for directory in REQUIRED_DIRS:
        exists = Path(directory).is_dir()
        print(f"[{'OK' if exists else 'FALHA'}] diretório {directory}")
        all_ok &= exists
    if not Path(".env").exists():
        print("[AVISO] .env não encontrado — copie de .env.example (opcional para execução local)")
    return all_ok


def main() -> int:
    """Executa todas as verificações e retorna código de saída."""
    ok = check_python() & check_packages() & check_structure()
    print("\nAmbiente válido!" if ok else "\nAmbiente inválido — corrija os itens acima.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
