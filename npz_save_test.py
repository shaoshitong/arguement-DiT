from PIL import Image
import numpy as np
from tqdm import tqdm

def create_npz_from_sample_folder(sample_dir, num=50_050):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        try:
            sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    samples = np.stack(samples)
    # assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

if __name__ == "__main__":
    # for i in range(3,6):
    #     j = 20000 + i * 40000
    #     create_npz_from_sample_folder(f"/mnt/weka/st_workspace/DDDM/arguement-DiT/samples_10k_dc_20250426_{j:07d}/SiT-XL-2-{j:07d}-size-256-vae-ema-cfg-1.5-seed-1")
    create_npz_from_sample_folder(f"/mnt/weka/st_workspace/DDDM/arguement-DiT/samples_10k_dc_20250426_0160000/SiT-XL-2-0160000-size-256-vae-ema-cfg-1.0-seed-1")
    