import subprocess
import time
import asyncio
import json
import cv2
import numpy as np

# Start the mothership server as a subprocess
server_proc = subprocess.Popen(
    ["python", "-m", "app.api.server"],
    cwd="C:\\Users\\YADVI\\OneDrive\\Desktop\\sih_model",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
print("Starting mothership server...")
time.sleep(3)

# Check if server started
if server_proc.poll() is not None:
    stdout, stderr = server_proc.communicate()
    print(f"Server failed to start!")
    print(f"stdout: {stdout.decode()}")
    print(f"stderr: {stderr.decode()}")
    exit(1)

print("Server started, PID:", server_proc.pid)

try:
    # Create client and test
    from app.network.client import MothershipClient
    
    async def test_mothership():
        client = MothershipClient(mothership_url='ws://localhost:8000', device_id='test_edge_device')
        
        # Create a test frame
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Send frame for inference
        result = await client.send_frame(frame)
        
        if result:
            print('Inference result:')
            print(f'  Status: {result["status"]}')
            print(f'  Confidence: {result["confidence"]*100:.1f}%')
            print(f'  Defect prob: {result["defect_prob"]*100:.4f}%')
            print(f'  Normal prob: {result["normal_prob"]*100:.4f}%')
        else:
            print('No inference result received')
        
        await client.disconnect()
    
    asyncio.run(test_mothership())
    
finally:
    # Terminate server
    print("Terminating server...")
    server_proc.terminate()
    server_proc.wait()
    print("Server stopped.")