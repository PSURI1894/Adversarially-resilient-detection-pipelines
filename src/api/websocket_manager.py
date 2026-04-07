"""
================================================================================
WEBSOCKET CONNECTION MANAGER
================================================================================
Manages concurrent WebSocket connections for real-time SOC dashboard.

Features:
    - Connection pool management for concurrent analysts
    - Topic-based message broadcasting (alerts, state, metrics)
    - Heartbeat and reconnection handling
    - Connection lifecycle tracking
================================================================================
"""

import json
import time
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Metadata about a connected client."""

    client_id: str
    connected_at: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(
        default_factory=lambda: {"alerts", "state", "metrics"}
    )
    last_heartbeat: float = field(default_factory=time.time)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time SOC dashboard.

    Supports topic-based filtering so clients can subscribe to
    specific message types (alerts, state changes, metrics).
    """

    def __init__(self, heartbeat_interval: float = 30.0):
        self._connections: Dict[str, Any] = {}  # client_id → websocket
        self._connection_info: Dict[str, ConnectionInfo] = {}
        self.heartbeat_interval = heartbeat_interval
        self._message_count = 0

    async def connect(
        self, websocket, client_id: str, subscriptions: Optional[Set[str]] = None
    ):
        """Register a new WebSocket connection."""
        await websocket.accept()
        self._connections[client_id] = websocket
        self._connection_info[client_id] = ConnectionInfo(
            client_id=client_id,
            subscriptions=subscriptions or {"alerts", "state", "metrics"},
        )
        logger.info(f"Client connected: {client_id} (total: {len(self._connections)})")

        # Send welcome message
        await self._send_to(
            client_id,
            {
                "type": "connected",
                "client_id": client_id,
                "timestamp": time.time(),
                "subscriptions": list(self._connection_info[client_id].subscriptions),
            },
        )

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        self._connections.pop(client_id, None)
        self._connection_info.pop(client_id, None)
        logger.info(
            f"Client disconnected: {client_id} (remaining: {len(self._connections)})"
        )

    async def broadcast(self, message: Dict[str, Any], topic: str = "alerts"):
        """
        Broadcast a message to all clients subscribed to the given topic.

        Parameters
        ----------
        message : dict
            The message payload.
        topic : str
            Message topic for filtering (alerts, state, metrics).
        """
        message["topic"] = topic
        message["timestamp"] = message.get("timestamp", time.time())
        self._message_count += 1

        disconnected = []
        for client_id, info in self._connection_info.items():
            if topic in info.subscriptions:
                try:
                    await self._send_to(client_id, message)
                except Exception:
                    disconnected.append(client_id)

        for cid in disconnected:
            await self.disconnect(cid)

    async def send_alert(self, alert: Dict[str, Any]):
        """Broadcast an alert event."""
        await self.broadcast(
            {
                "type": "alert",
                "data": alert,
            },
            topic="alerts",
        )

    async def send_state_update(self, state: Dict[str, Any]):
        """Broadcast a SOC state change."""
        await self.broadcast(
            {
                "type": "state_update",
                "data": state,
            },
            topic="state",
        )

    async def send_metrics(self, metrics: Dict[str, Any]):
        """Broadcast performance metrics."""
        await self.broadcast(
            {
                "type": "metrics",
                "data": metrics,
            },
            topic="metrics",
        )

    async def _send_to(self, client_id: str, message: Dict):
        """Send a message to a specific client."""
        ws = self._connections.get(client_id)
        if ws:
            await ws.send_json(message)

    async def handle_client_message(self, client_id: str, data: str):
        """
        Process an incoming message from a client.
        Supports: subscribe, unsubscribe, heartbeat.
        """
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")
        info = self._connection_info.get(client_id)
        if not info:
            return

        if msg_type == "subscribe":
            topics = msg.get("topics", [])
            info.subscriptions.update(topics)
        elif msg_type == "unsubscribe":
            topics = msg.get("topics", [])
            info.subscriptions -= set(topics)
        elif msg_type == "heartbeat":
            info.last_heartbeat = time.time()
            await self._send_to(client_id, {"type": "heartbeat_ack"})

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    @property
    def total_messages(self) -> int:
        return self._message_count

    def get_connection_stats(self) -> Dict[str, Any]:
        """Return connection pool statistics."""
        return {
            "active_connections": self.active_connections,
            "total_messages_sent": self._message_count,
            "clients": [
                {
                    "client_id": info.client_id,
                    "connected_at": info.connected_at,
                    "subscriptions": list(info.subscriptions),
                    "last_heartbeat": info.last_heartbeat,
                }
                for info in self._connection_info.values()
            ],
        }
