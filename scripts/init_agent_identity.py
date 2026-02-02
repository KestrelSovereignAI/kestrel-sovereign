import sys
import os

# Add the app directory to the path so we can import modules
sys.path.insert(0, '/app')

try:
    from kestrel_sovereign.inception_service import create_kestrel_identity
    
    # Ensure /app directory exists (it should in the container)
    target_dir = '/app'
    if not os.path.exists(target_dir):
        # Fallback for local testing if /app doesn't exist
        target_dir = os.getcwd()
        
    creds = create_kestrel_identity(target_dir)
    print(f'Created agent: {creds.agent_did}')
    print(f'Database: {creds.db_path}')
except ImportError:
    print("Error: Could not import inception_service. Make sure you are running this from the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f'Error creating agent: {e}')
    sys.exit(1)
