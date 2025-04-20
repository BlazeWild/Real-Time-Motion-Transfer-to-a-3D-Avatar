import asyncio
import json
import websockets
import os

# Global variable to store the latest keypoints data
latest_keypoints = {}

# Set to store connected clients
connected_clients = set()

async def register(websocket):
    """Register a new client connection"""
    print(f"New client connected")
    connected_clients.add(websocket)
    try:
        # Keep the connection alive
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected")

async def broadcast_keypoints():
    """Broadcast keypoints to all connected clients"""
    while True:
        if connected_clients and latest_keypoints:  # Only broadcast if we have clients and data
            # Convert to JSON string
            message = json.dumps(latest_keypoints)
            # Send to all connected clients
            websockets.broadcast(connected_clients, message)
        # Short delay to avoid sending too frequently
        await asyncio.sleep(0.05)  # 50ms delay

async def main():
    """Start the WebSocket server"""
    print("Starting WebSocket server...")
    
    # Start the server
    async with websockets.serve(register, "localhost", 8765):
        # Start broadcasting task
        broadcast_task = asyncio.create_task(broadcast_keypoints())
        
        print("WebSocket server running at ws://localhost:8765")
        print("Waiting for keypoints data and client connections...")
        
        # Keep the server running
        await asyncio.Future()

# Function to update keypoints from external code
def update_keypoints(data):
    """Update the latest keypoints data"""
    global latest_keypoints
    latest_keypoints = data

# If running directly, start the server
if __name__ == "__main__":
    asyncio.run(main()) 