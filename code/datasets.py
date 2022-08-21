# Some parts of this code is borrowed from https://github.com/xahidbuffon/FUnIE-GAN
import glob,json,random
import numpy as np
import torch
from PIL import Image

# norm and denorm the image
def norm(image):
    return (image / 127.5) - 1.0

def denorm(image):
    return (image + 1.0) * 127.5

def augment(dt, eh):
    a = random.random()
    dt = dt * a + eh * (1 - a)
    if random.random() < 0.25:
        dt = np.fliplr(dt)
        eh = np.fliplr(eh)
    if random.random() < 0.25:
        dt = np.flipud(dt)
        eh = np.flipud(eh)

    return dt, eh
    
#dataset for unpair data    
class Unpairdata(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size, split):
        super(Unpairdata, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.split = split

        # Load JSON of splits
        names = json.load(open(f"{self.data_root}/splits.json", "r"))[self.split]

        # Build image paths
        self.dt_ims = [f"{self.data_root}/{n}" for n in names if "trainA" in n]
        self.eh_ims = [f"{self.data_root}/{n}" for n in names if "trainB" in n]
        print(f"Total {len(self.dt_ims)} poor quality data")
        print(f"Total {len(self.eh_ims)} good quality data")

        # Force # of images to the least amount
        num = min(len(self.dt_ims), len(self.eh_ims))
        self.dt_ims = self.dt_ims[:num]
        self.eh_ims = self.eh_ims[:num]
        print(f"Total {len(self.eh_ims)} data used")

    def __getitem__(self, index):
        # Read and resize image pair
        dt = Image.open(self.dt_ims[index]).convert("RGB")
        eh = Image.open(self.eh_ims[index]).convert("RGB")
        dt = dt.resize(self.im_size)
        eh = eh.resize(self.im_size)

        # Transfrom image pair to float32 np.ndarray
        dt = np.array(dt, dtype=np.float32)
        eh = np.array(eh, dtype=np.float32)

        # Augment image pair
        if self.split == "train":
            dt, eh = augment(dt, eh)

        # Transfrom image pair to (C, H, W) torch.Tensor
        dt = torch.Tensor(norm(dt)).permute(2, 0, 1)
        eh = torch.Tensor(norm(eh)).permute(2, 0, 1)
        return dt, eh

    def __len__(self):
        return len(self.dt_ims)
# dataset for pair data
class Pairdata(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size, split):
        super(Pairdata, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.split = split
        names = json.load(open(f"{self.data_root}/splits.json", "r"))[self.split]
        self.dt_ims = [f"{self.data_root}/{n}" for n in names]
        self.eh_ims = [f"{self.data_root}/{n}" for n in names]
        print(f"Total {len(self.dt_ims)} data")
# Read and resize image pair
    def __getitem__(self, index):
        
        dt = Image.open(self.dt_ims[index]).convert("RGB")
        eh = Image.open(self.eh_ims[index]).convert("RGB")
        dt = dt.resize(self.im_size)
        eh = eh.resize(self.im_size)
        dt = np.array(dt, dtype=np.float32)
        eh = np.array(eh, dtype=np.float32)
        if self.split == "train":
            dt, eh = augment(dt, eh)
        dt = torch.Tensor(norm(dt)).permute(2, 0, 1)
        eh = torch.Tensor(norm(eh)).permute(2, 0, 1)
        return dt, eh

    def __len__(self):
        return len(self.dt_ims)

# dataset for test data
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, im_size):
        super(TestDataset, self).__init__()

        self.data_root = data_root
        self.im_size = im_size
        self.ims = glob.glob(f"{self.data_root}/*")

    def __getitem__(self, index):
        # Read and resize image
        path = self.ims[index]
        im = Image.open(path).convert("RGB")
        im = im.resize(self.im_size)

        # Transfrom image to float32 np.ndarray
        im = np.array(im, dtype=np.float32)

        # Transfrom image to (C, H, W) torch.Tensor
        im = torch.Tensor(norm(im)).permute(2, 0, 1)
        return path, im

    def __len__(self):
        return len(self.ims)


if __name__ == "__main__":
    dataset = Pairdata(
        data_root="../data/EUVP Dataset/Paired/underwater_dark", im_size=(256, 256))
    image, target = dataset[0]
