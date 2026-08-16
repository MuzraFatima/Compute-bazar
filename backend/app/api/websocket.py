"""
=============================================================
 COMPUTE BAZAAR — WebSocket Connection Manager (Weeks 3-8)
=============================================================
 Broadcaster class that handles client socket subscriptions
 and pushes real-time updates when bids are submitted
 or when auction rounds are solved.
=============================================================
"""

from typing import List
from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """Manages active WebSockets clients for live dashboard push updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Register a new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Deregister a connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        """Broadcast JSON message to all connected clients."""
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                # Connection might have closed without clean handshake
                logger.warning(f"Error sending WebSocket message: {e}")
                disconnected_clients.append(connection)

        # Cleanup inactive connections
        for client in disconnected_clients:
            self.disconnect(client)


# Global instance
manager = ConnectionManager()
