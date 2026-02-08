import os
import requests
import sqlite3
import sys

from kestrel_sovereign.kestrel_config.constants import HTTP_TIMEOUT_SHORT
from kestrel_sovereign.kestrel_config.defaults import get_ollama_url

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
        else:
            print(f"Ollama: ❌ Responded with status {resp.status_code}")
    except requests.exceptions.RequestException:
        print(f'Ollama: ❌ Not responding at {ollama_url}')

    # Check config
    config_exists = os.path.exists('llm_config.toml')
    print(f'LLM Config: {"✅ Found" if config_exists else "❌ Missing (run `cp llm_config.toml.example llm_config.toml`)"}')

    print('\n🚀 Ready to start!')

if __name__ == "__main__":
    run_health_check() 