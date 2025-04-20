import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import time

# Set to store connected clients
connected_clients = set()

# Latest frame data (base64 encoded)
latest_frame = None
latest_frame_time = 0

async def register(websocket):
    """Register a new client connection"""
    print(f"New video client connected")
    connected_clients.add(websocket)
    try:
        # Send initial frame if available
        if latest_frame:
            frame_data = {
                "image": latest_frame,
                "timestamp": latest_frame_time
            }
            await websocket.send(json.dumps(frame_data))
            print(f"Sent initial frame to new client")
        
        # Keep the connection alive
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print(f"Video client disconnected")

async def broadcast_frames():
    """Broadcast video frames to all connected clients"""
    global latest_frame_time
    
    broadcast_count = 0
    while True:
        if connected_clients and latest_frame:  # Only broadcast if we have clients and a frame
            try:
                # Create data packet with frame
                frame_data = {
                    "image": latest_frame,
                    "timestamp": latest_frame_time
                }
                
                # Convert to JSON string
                message = json.dumps(frame_data)
                
                # Send to all connected clients
                await asyncio.gather(
                    *[client.send(message) for client in connected_clients],
                    return_exceptions=True
                )
                
                broadcast_count += 1
                if broadcast_count % 100 == 0:
                    print(f"Video broadcast stats: {broadcast_count} total broadcasts, {len(connected_clients)} clients")
            except Exception as e:
                print(f"Error during video broadcast: {e}")
        
        # Short delay to avoid sending too frequently (30fps target)
        await asyncio.sleep(0.033)  # ~30fps

async def main():
    """Start the WebSocket server for video frames"""
    print("Starting Video WebSocket server...")
    
    # Start the server
    async with websockets.serve(register, "localhost", 8766):
        # Start broadcasting task
        broadcast_task = asyncio.create_task(broadcast_frames())
        
        print("Video WebSocket server running at ws://localhost:8766")
        print("Waiting for video frames and client connections...")
        
        # Keep the server running
        await asyncio.Future()

def update_frame(frame):
    """Update the latest frame from external code
    
    Args:
        frame: A numpy array (BGR) representing the frame
    """
    global latest_frame, latest_frame_time
    
    try:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Convert to base64 for sending over WebSocket
        latest_frame = base64.b64encode(buffer).decode('utf-8')
        latest_frame_time = time.time()
    except Exception as e:
        print(f"Error encoding video frame: {e}")

# If running directly, start the server
if __name__ == "__main__":
    asyncio.run(main()) 