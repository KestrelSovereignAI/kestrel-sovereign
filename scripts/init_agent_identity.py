import sys
import os
from pathlib import Path

# Add the app directory to the path so we can import modules
sys.path.insert(0, '/app')

try:
    from kestrel_sovereign.inception_service import create_kestrel_identity
    
    target_dir = Path(os.environ.get("KESTREL_DB_PATH", "/app/agent_data"))
    if target_dir.is_absolute() and not target_dir.parent.exists():
        # Fallback for local testing if /app doesn't exist.
        target_dir = Path.cwd() / "agent_data"
    target_dir.mkdir(parents=True, exist_ok=True)
        
    creds = create_kestrel_identity(str(target_dir))
    print(f'Created agent: {creds.agent_did}')
    print(f'Database: {creds.db_path}')
except ImportError:
    print("Error: Could not import inception_service. Make sure you are running this from the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f'Error creating agent: {e}')
    sys.exit(1)
