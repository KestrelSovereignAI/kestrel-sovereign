# Kestrel API Endpoints
from fastapi import APIRouter

# Import all routers
from .agent import router as agent_router
from .conversations import router as conversations_router
from .memories import router as memories_router
from .sovereignty import router as sovereignty_router
from .database import router as database_router
from .models import router as models_router
from .security import router as security_router
from .commands import router as commands_router
from .files import router as files_router
from .observability import router as observability_router
from .saved_items import router as saved_items_router
from .metrics import router as metrics_router
from .spawn import router as spawn_router
from .voice import router as voice_router

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
    "observability_router",
    "saved_items_router",
    "metrics_router",
    "spawn_router",
    "voice_router",
]
