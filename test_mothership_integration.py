import asyncio
import json
import base64
import cv2
import numpy as np
import websockets
from app.network.client import MothershipClient


async def test_mothership():
    # Create client
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


async def test_continuous_stream():
    """Test continuous streaming mode."""
    frame_count = [0]

    async def get_frame():
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame_count[0] += 1
        return (frame, 0.0)

    async def should_stop():
        return frame_count[0] >= 3

    client = MothershipClient(mothership_url='ws://localhost:8000', device_id='test_continuous_device')

    print("Starting continuous stream test (3 frames)...")
    results = []
    async for result in client.continuous_stream(
        frame_callback=get_frame,
        stop_callback=should_stop,
        max_frames=3
    ):
        results.append(result)
        print(f"  Frame {len(results)}: Status={result['status']}, Confidence={result['confidence']*100:.1f}%")

    await client.disconnect()
    print(f"Processed {len(results)} frames total")


async def main():
    print("=" * 50)
    print("Test 1: Single frame inference")
    print("=" * 50)
    await test_mothership()

    print()
    print("=" * 50)
    print("Test 2: Continuous streaming")
    print("=" * 50)
    await test_continuous_stream()

    print()
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())