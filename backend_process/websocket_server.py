import asyncio
import json
import websockets
import os
import time

# Global variable to store the latest keypoints data
latest_keypoints = {}

# Set to store connected clients
connected_clients = set()

# Debug info
broadcast_count = 0
last_update_time = 0

async def register(websocket):
    """Register a new client connection"""
    print(f"New client connected")
    connected_clients.add(websocket)
    try:
        # Send initial data immediately if available
        if latest_keypoints:
            await websocket.send(json.dumps(latest_keypoints))
            print(f"Sent initial data to new client with {len(latest_keypoints)} keypoints")
        
        # Keep the connection alive
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected")

async def broadcast_keypoints():
    """Broadcast keypoints to all connected clients"""
    global broadcast_count, last_update_time
    
    while True:
        if connected_clients and latest_keypoints:  # Only broadcast if we have clients and data
            try:
                # Convert to JSON string
                message = json.dumps(latest_keypoints)
                
                # Send to all connected clients
                await asyncio.gather(
                    *[client.send(message) for client in connected_clients],
                    return_exceptions=True
                )
                
                broadcast_count += 1
                if broadcast_count % 100 == 0:
                    print(f"Broadcast stats: {broadcast_count} total broadcasts, {len(connected_clients)} clients")
                    print(f"Time since last update: {time.time() - last_update_time:.3f}s")
                    print(f"Keypoint count: {len(latest_keypoints)}")
            except Exception as e:
                print(f"Error during broadcast: {e}")
        
        # Short delay to avoid sending too frequently
        await asyncio.sleep(0.02)  # 20ms delay (50Hz)

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
    global latest_keypoints, last_update_time
    latest_keypoints = data
    last_update_time = time.time()

# If running directly, start the server
if __name__ == "__main__":
    asyncio.run(main()) 