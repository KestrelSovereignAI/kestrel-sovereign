import os
import requests
import sqlite3
import sys

from kestrel_sovereign.config import load_section
from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_SHORT
from kestrel_sovereign.kestrel_config.defaults import get_ollama_url
from kestrel_sovereign.llm.model_selection import resolve_provider_default


def _resolve_expected_ollama_model(installed_models: list[str]) -> str | None:
    """Resolve the Ollama model Kestrel is configured to use.

    Reads the route config under ``[llm.vendors.ollama.routes.local]`` (the
    vendor/route schema established by #688). The pre-#688 flat ``[ollama]``
    block is no longer recognised by anything else in the system; reading it
    here would silently return stale data on configs that have been migrated.
    """
    llm_config = load_section("llm")
    ollama_route = (
        llm_config
        .get("vendors", {})
        .get("ollama", {})
        .get("routes", {})
        .get("local", {})
    ) or {}
    configured_model = ollama_route.get("model")

    if configured_model and configured_model != "auto":
        return str(configured_model)

    try:
        return resolve_provider_default("ollama", llm_config=llm_config)
    except Exception:
        selection_hints = ollama_route.get("selection_hints", []) or []
        for hint in selection_hints:
            hint_lower = str(hint).lower()
            for model_id in installed_models:
                if hint_lower in model_id.lower():
                    return model_id

    if len(installed_models) == 1:
        return installed_models[0]

    return None

def run_health_check():
    """
    Runs a series of checks to verify the Kestrel development environment.
    """
    print('🔍 Kestrel Health Check')
    print('=' * 30)

    # Check environment
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working dir: {os.getcwd()}")

    # Check databases
    try:
        dbs = [f for f in os.listdir('.') if f.endswith('.db')]
        print(f"Agent databases: {len(dbs)} found")
        for db in dbs[:3]:  # Show first 3
            print(f'  - {db}')
    except FileNotFoundError:
        print("Agent databases: No database files found.")

    # Check Ollama
    try:
        ollama_url = get_ollama_url()
        resp = requests.get(f'{ollama_url}/api/tags', timeout=HTTP_TIMEOUT_SHORT)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            print(f"Ollama: ✅ Running ({len(models)} models)")
            installed_model_ids = [model.get('name', '') for model in models if model.get('name')]
            expected_model = _resolve_expected_ollama_model(installed_model_ids)

            if expected_model:
                if expected_model in installed_model_ids:
                    print(f"  Configured model: ✅ {expected_model}")
                else:
                    print(f"  Configured model: ⚠️  {expected_model} is not installed")
                    print(f"  Run: ollama pull {expected_model}")
                    if installed_model_ids:
                        print(f"  Installed models: {', '.join(installed_model_ids[:5])}")
            elif installed_model_ids:
                print("  Configured model: ⚠️  Unable to resolve active Ollama model from kestrel.toml [llm]")
        else:
            print(f"Ollama: ❌ Responded with status {resp.status_code}")
    except requests.exceptions.RequestException:
        print(f'Ollama: ❌ Not responding at {ollama_url}')

    # Check config
    from kestrel_sovereign.paths import project_dir
    kestrel_toml_path = project_dir() / "kestrel.toml"
    kestrel_toml_exists = kestrel_toml_path.exists()
    if kestrel_toml_exists:
        llm_section = load_section("llm")
        if llm_section:
            print(f'LLM Config: ✅ {kestrel_toml_path} [llm] populated')
        else:
            print(f'LLM Config: ⚠️  {kestrel_toml_path} found but [llm] section is empty')
    else:
        print(f'LLM Config: ❌ Missing {kestrel_toml_path} — run `kestrel setup` to create one')

    print('\n🚀 Ready to start!')

if __name__ == "__main__":
    run_health_check() 