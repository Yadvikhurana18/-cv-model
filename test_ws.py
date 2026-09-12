"""Test WebSocket connection to mothership server."""
import asyncio
import base64
import numpy as np
import cv2
import websockets
import time


async def test_ws():
    # Connect to the mothership WS
    ws = await websockets.connect('ws://localhost:8000/ws/infer?device_id=test-device')

    # Create a test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', test_img)
    frame_b64 = base64.b64encode(buf).decode('utf-8')

    # Send frame
    msg = {
        'type': 'frame',
        'device_id': 'test-device',
        'frame_base64': frame_b64,
        'timestamp': time.time()
    }
    await ws.send(str(msg))
    print('[WS] Sent frame')

    # Receive result
    response = await ws.recv()
    print('[WS] Received:', response)

    await ws.close()


if __name__ == "__main__":
    asyncio.run(test_ws())