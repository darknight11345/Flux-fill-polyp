import pickle
import matplotlib.pyplot as plt
import numpy as np

pkl_path="/pfs/work9/workspace/scratch/ul_swv79-thesis_flw/diffuse-gen/sampling_images/cluster1-diffgen-2026-08-18-11-53/styled_samples_1x512x512x4.pkl"

#"/pfs/work9/workspace/scratch/ul_swv79-thesis_flw/diffuse-gen/sampling_images/cluster1-diffgen-2026-08-17-21-50/samples_1x512x512x3.pkl"




with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())

# Take the first style/path
key = next(iter(data))
images = data[key]

print("Key:", key)
print("Number of images:", len(images))
print("Image shape:", images[0].shape)
print("Image dtype:", images[0].dtype)

img = images[0]

if img.shape[-1] == 4:
    rgb = img[..., :3]
    mask = img[..., 3]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    plt.show()

else:
    plt.imshow(img)
    plt.axis("off")
    plt.show()