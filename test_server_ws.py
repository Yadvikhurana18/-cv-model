"""Comprehensive test: start mothership server + WebSocket inference test."""
import asyncio
import base64
import numpy as np
import cv2
import websockets
import time
import subprocess
import sys
import os
import signal


async def test_ws():
    """Test WebSocket connection to mothership."""
    # Wait for server to be ready
    await asyncio.sleep(2)
    
    try:
        ws = await websockets.connect('ws://localhost:8000/ws/infer?device_id=test-device')
        print('[WS] Connected to mothership!')
    except ConnectionRefusedError:
        print('[WS] Connection refused - server not ready')
        return False

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
    try:
        response = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print('[WS] Received:', response)
        await ws.close()
        print('[WS] Test completed successfully!')
        return True
    except asyncio.TimeoutError:
        print('[WS] Timeout waiting for response')
        await ws.close()
        return False


def main():
    # Start server in background
    print('[*] Starting mothership server...')
    proc = subprocess.Popen(
        [sys.executable, 'run.py', '--mode', 'api', '--port', '8000'],
        cwd='C:\\Users\\YADVI\\OneDrive\\Desktop\\sih_model',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print('[*] Waiting for server to start (3s)...')
    time.sleep(3)
    
    # Check if process is still running
    if proc.poll() is not None:
        print('[!] Server process exited unexpectedly')
        stderr = proc.stderr.read().decode('utf-8', errors='replace')
        print('[!] Error output:', stderr[:500])
        return
    
    # Run WS test
    print('[*] Running WebSocket test...')
    success = asyncio.run(test_ws())
    
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