# Kestrel API Endpoints
from fastapi import APIRouter

# Core routers (always present, not feature-gated)
from .agent import router as agent_router
from .conversations import router as conversations_router
from .memories import router as memories_router
from .sovereignty import router as sovereignty_router
from .database import router as database_router
from .models import router as models_router
from .security import router as security_router
from .commands import router as commands_router
from .files import router as files_router
from .saved_items import router as saved_items_router
from .metrics import router as metrics_router

# Feature-contributed routers (voice, spawn, observability) are no longer
# imported here. They are mounted dynamically via Feature.get_router()
# during server startup. See server.py _mount_feature_routers().

__all__ = [
    "agent_router",
    "conversations_router",
    "memories_router",
    "sovereignty_router",
    "database_router",
    "models_router",
    "security_router",
    "commands_router",
    "files_router",
    "saved_items_router",
    "metrics_router",
]
