"""Test mothership server with WebSocket inference."""
import asyncio
import base64
import json
import numpy as np
import cv2
import websockets
import subprocess
import sys
import os
import time


async def test_ws_connection():
    """Test WebSocket connection to mothership."""
    await asyncio.sleep(3)
    
    try:
        ws = await websockets.connect('ws://localhost:8003/ws/infer?device_id=test-device')
        print('[WS] Connected to mothership!')
    except ConnectionRefusedError:
        print('[WS] Connection refused - server not ready')
        return False

    # Create a test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', test_img)
    frame_b64 = base64.b64encode(buf).decode('utf-8')

    # Send frame as proper JSON
    msg = json.dumps({
        'type': 'frame',
        'device_id': 'test-device',
        'frame_base64': frame_b64,
        'timestamp': time.time()
    })
    await ws.send(msg)
    print('[WS] Sent JSON frame with base64 image')

    # Receive result
    try:
        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print('[WS] Received inference result:', json.dumps(response, indent=2)[:200])
        await ws.close()
        print('[WS] Test completed successfully!')
        return True
    except asyncio.TimeoutError:
        print('[WS] Timeout waiting for response')
        await ws.close()
        return False
    except Exception as e:
        print('[WS] Error receiving response:', e)
        await ws.close()
        return False


def main():
    # Start mothership server in background
    print('[*] Starting mothership server on port 8003...')
    proc = subprocess.Popen(
        [sys.executable, 'run.py', '--mode', 'api', '--port', '8003'],
        cwd='C:\\Users\\YADVI\\OneDrive\\Desktop\\sih_model',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    # Wait for server to start
    print('[*] Waiting for server to initialize (5s)...')
    time.sleep(5)
    
    # Check if process is still running
    if proc.poll() is not None:
        print('[!] Server process exited unexpectedly')
        stderr = proc.stderr.read().decode('utf-8', errors='replace')
        print('[!] Error output:', stderr[:500])
        return
    
    # Run WS test
    print('[*] Running WebSocket inference test...')
    success = asyncio.run(test_ws_connection())
    
    # Terminate server
    print('[*] Terminating server...')
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    
    if success:
        print('[✓] Mothership WebSocket test PASSED')
    else:
        print('[✗] Mothership WebSocket test FAILED')


if __name__ == "__main__":
    main()