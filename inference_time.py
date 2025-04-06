import time
import numpy as np

model.eval()
timings = []

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        start = time.time()
        outputs = model(images)
        end = time.time()
        timings.append((end - start) / images.size(0))  # Time per image

avg_inference_time = np.mean(timings)
print(f"Average Inference Time per Image: {avg_inference_time * 1000:.2f} ms")
