"""
================================================================================
SOC DASHBOARD API — FASTAPI + WEBSOCKET BACKEND
================================================================================
"""

from .server import create_app
from .websocket_manager import WebSocketManager

__all__ = ["create_app", "WebSocketManager"]
