import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from diffusion import create_diffusion
from dataset.generate_argument import prepare_images

class ImageSampler:
    def __init__(self, model="DiT-XL/2", vae="mse", image_size=256, num_classes=1000, num_sampling_steps=250, cfg_scale=1.5, seed=0, ckpt=None):
        """
        Initialize the ImageSampler with the specified parameters.
        """
        self.model_name = model
        self.vae = vae
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.ckpt = ckpt or f"DiT-XL-2-{self.image_size}x{self.image_size}.pt"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup PyTorch
        torch.manual_seed(self.seed)
        
        # Load model
        latent_size = self.image_size // 8
        self.model = DiT_models[self.model_name](
            input_size=latent_size,
            num_classes=self.num_classes
        ).to(self.device)
        
        state_dict = find_model(self.ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Create diffusion
        self.diffusion = create_diffusion(str(self.num_sampling_steps))

    def _center_crop_arr(self, pil_image, image_size):
        """
        Center crop image to match the desired size
        """
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    def _get_image(self, img_path, transform):
        """
        Load image and apply transformation
        """
        lodder = default_loader
        img = lodder(img_path)
        image = transform(img).unsqueeze(0)
        return image

    def _prepare_image(self, image_path, mix_image_path):
        """
        Preprocess the image and mix image to match the model input
        """
        prepare_img = prepare_images(
            Image.open(image_path).convert("RGB"), 
            Image.open(mix_image_path).convert("RGB"), 
            self.image_size, 
            5
        )[0]
        prepare_img = prepare_img * 2 - 1
        return prepare_img

    def sample_images(self, images, class_labels,device,vae):
        """
        Sample images using the model from a batch of images and their corresponding cutmix images.
        """
        # Process the batch of images
        


        images_tensor = images

        x = vae.encode(images_tensor.to(device)).latent_dist.sample().mul_(0.18215)

        sampled_images = []

        t = 40
        save_t = t
        t = torch.tensor([0] * x.shape[0], device=self.device)
        n = len(class_labels)
        y_null = torch.tensor([1000] * n, device=self.device)
        
        model_output = self.model.forward(x, t, y_null)
        model_output, model_var_values = torch.split(model_output, 4, dim=1)

        x_t = self.diffusion.q_sample(x, torch.tensor([save_t] * x.shape[0], device=self.device), noise=model_output)
        z = x_t
        y = torch.tensor(class_labels, device=self.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=self.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=self.cfg_scale)

        # Sample images:
        samples = self.diffusion.p_sample_loop_with_t(
            self.model.forward_with_cfg, z.shape, z,
            clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=self.device, begin_t=save_t
        )

        samples_return, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # samples = vae.decode(samples_return / 0.18215).sample
        # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        # os.makedirs("samples_refine", exist_ok=True)

        # for i, sample in enumerate(samples):
        #     image_filename = os.path.join("samples_refine", f"{int(i):04d}.png")
        #     Image.fromarray(sample).save(image_filename)
        #     print(f"Saved image: {image_filename}")      



        # Collect the sampled images

        return samples_return


# Example usage:

def main():
    images = ["path_to_image1.jpg", "path_to_image2.jpg"]  # List of image paths
    cutmix_images = ["path_to_cutmix_image1.jpg", "path_to_cutmix_image2.jpg"]  # List of cutmix image paths
    class_labels = [22, 30]  # Example class labels for the images

    sampler = ImageSampler(model="DiT-XL/2", image_size=256, num_classes=1000, num_sampling_steps=250, cfg_scale=1.5)
    sampled_images = sampler.sample_images(images, cutmix_images, class_labels)

    # Save the sampled images
    for i, img in enumerate(sampled_images):
        Image.fromarray(img).save(f"sampled_image_{i}.png")
        print(f"Saved image {i}.")

if __name__ == "__main__":
    main()
