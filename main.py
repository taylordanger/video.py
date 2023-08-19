import os
import cv2
from os.path import isfile, join
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import shutil

learning_rate = 0.0002
num_epochs = 100
batch_size = 128
video_directory = '/path/to/the/videos/'
video_paths = [os.path.join(video_directory, video) for video in os.listdir(video_directory) if video.endswith(".mp4")]
output_directory = '/to/path/the/output_images/'
new_video_output_path = '/path/to/the/output.mp4'
z_dim = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=3, input_size=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 1024, 1, 1)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalization=False),
            *discriminator_block(64, 128),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)

        )

        self.flatten = nn.Sequential(
            nn.Linear(256 * 5 * 5, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        feature = self.model(x)
        feature = feature.view(feature.size(0), -1)  # This reshapes the tensor
        out = self.flatten(feature)
        return out

generator = Generator(input_dim=100, output_dim=3, input_size=64).to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

visual_epoch_interval = 10  # Adjust this value based on how often you want to see the output

try:
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        for real_batch, _ in tqdm(dataloader, desc="Batches", leave=False):
            real_batch = real_batch.to(device)

            current_batch_size = real_batch.size(0)
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            # Discriminator train
            d_optimizer.zero_grad()

            # Real data train
            real_output = discriminator(real_batch)
            # 2. Ensure labels have the same size as the output
            real_labels = real_labels[:real_output.size(0)]
            d_real_loss = criterion(real_output, real_labels)

            # Fake data train
            noise_vectors = torch.randn(current_batch_size, z_dim).to(device)
            fake_images = generator(noise_vectors)
            fake_output = discriminator(fake_images.detach())
            # Ensure labels have the same size as the output
            fake_labels = fake_labels[:fake_output.size(0)]
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Generator train
            g_optimizer.zero_grad()
            gen_output = discriminator(fake_images)
            # Ensure labels have the same size as the output
            real_labels = real_labels[:gen_output.size(0)]
            g_loss = criterion(gen_output, real_labels)
            g_loss.backward()
            g_optimizer.step()



            print(f"Epoch [{epoch + 1}/{num_epochs}] d_loss: {d_loss.item()} g_loss: {g_loss.item()}")
except Exception as e:
    print(f"Error during training: {e}")

#  trans
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def video_to_frames(video, path_output_dir):
    """Function to split video into frames."""

    try:
        vidcap = cv2.VideoCapture(video)
    except Exception as e:
        print(f"Failed to open video file {video}. Error: {e}")
        return

    # If the video file could not be opened, print an error and return.
    if not vidcap.isOpened():
        print(f"Could not open video file {video}.")
        return

    # Attempt to create the output directory.
    try:
        os.makedirs(path_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory {path_output_dir}. Error: {e}")
        return

    count = 0
    while vidcap.isOpened():
        try:
            success, image = vidcap.read()
        except Exception as e:
            print(f"Failed to read frame from video file {video}. Error: {e}")
            break

        if success:
            try:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
            except Exception as e:
                print(f"Failed to write frame to file. Error: {e}")
                break
            count += 1
        else:
            break

    # Release the video file and destroy all windows.
    vidcap.release()
    cv2.destroyAllWindows()

def frames_to_video(inputpath, outputpath, fps):
    """Convert frames back to video."""
    image_array = []
    files = [f for f in os.listdir(inputpath) if isfile(join(inputpath, f))]
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        img = cv2.imread(inputpath + files[i])
        size =  (img.shape[1],img.shape[0])
        img = cv2.resize(img,size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath,fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()

def process_frame_through_gan(generator):
    """Process frame through the GAN."""
    noise_tensor = torch.randn(1, 1024, 1, 1).narrow(2, 1, 100)
    generated_image = generator(noise_tensor)
    generated_image = generated_image.squeeze(0)  # Remove the batch dimension
    return transforms.ToPILImage()(generated_image.cpu())

def clear_directory(directory):
    """Clear files in a directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

fixed_noise = torch.randn(64, z_dim, 1, 1).to(device) # This will be used to visualize the progression of the generator

def visualize_generated_images(generator, noise, epoch):
    """Visualize generated images."""
    with torch.no_grad():
        generated = generator(noise).detach().cpu()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.imshow(vutils.make_grid(generated, padding=2, normalize=True).permute(1, 2, 0))
    plt.show()

# Processing
for video_path in tqdm(video_paths, desc="Processing Videos"):
    video_to_frames(video_path, output_directory)
    frame_files = sorted(os.listdir(output_directory), key=lambda x: int(x.split('.')[0]))
    for frame in tqdm(frame_files, desc="Frames", leave=False):
        frame_path = os.path.join(output_directory, frame)
        generator.eval()
        generated_image = process_frame_through_gan(generator)
        generator.train()
        generated_image_path = os.path.join(output_directory, f"gen_{frame}")
        generated_image.save(generated_image_path)
