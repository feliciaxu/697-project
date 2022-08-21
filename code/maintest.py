# Some parts of this code is borrowed from https://github.com/xahidbuffon/FUnIE-GAN
import argparse,os,time,torch
import numpy as np
from datasets import TestDataset, denorm
from models import Ver1, Ver2, Gene
from torchvision import transforms
from utils import AverageMeter, ProgressMeter

# this is the main file
class Predictor(object):
    def __init__(self, model, test_loader, mpath, spath, is_cuda):

        self.test_loader = test_loader
        self.spath = spath
        os.makedirs(self.spath, exist_ok=True)

        self.is_cuda = is_cuda
        self.print_freq = 20

        self.model = model
        if not os.path.isfile(mpath):
            raise FileNotFoundError(f"The file '{mpath}' not found!")
        self.load(mpath)
        if self.is_cuda:
            self.model.cuda()

    def predict(self):
        self.model.eval()

        batch_time = AverageMeter("Time", "3.3f")
        progress = ProgressMeter(len(self.test_loader), [
                                 batch_time], prefix="Test: ")

        with torch.no_grad():
            end = time.time()
            for batch_idx, (paths, images) in enumerate(self.test_loader):
                bs = images.size(0)
                if self.is_cuda:
                    images = images.cuda()
                Fim = self.model(images)

                Fim = denorm(Fim.data)
                Fim = torch.clamp(Fim, min=0., max=255.)
                Fim = Fim.type(torch.uint8)

                for idx in range(bs):
                    name = os.path.splitext(os.path.basename(paths[idx]))[0]
                    fake_image = Fim[idx]
                    fake_image = transforms.ToPILImage()(fake_image).convert("RGB")
                    fake_image.save(f"{self.spath}/{name}.png")

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % self.print_freq == 0:
                    progress.display(batch_idx)
        return

    def load(self, model):
        device = "cuda:0" if self.is_cuda else "cpu"
        ckpt = torch.load(model, map_location=device)
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"We are in epoch: {ckpt['epoch']} (loss={ckpt['best_loss']:.3f})")
        print(f">>>We Load generator from {model}")


if __name__ == "__main__":

    # try different models on diffrent versions
    np.random.seed(77)
    torch.manual_seed(77)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(77)

    vername = ["version1", "version2", "unpair"]
    Varch = [Ver1, Ver2, Gene]
    model_mapper = {m: net for m, net in zip(vername, Varch)}

    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("-d", "--data", default="../mix/trainA", type=str, metavar="PATH",
                        help="path to data (default: none)")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="version1",
                        choices=vername,
                        help="choose version")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers")
    parser.add_argument("-b", "--batch-size", default=10, type=int,
                        metavar="N",
                        help="this is batch-size")
    parser.add_argument("-m", "--model", default="result/best_gen.pth.tar", type=str, metavar="PATH",
                        help="check the path of gen")
    parser.add_argument("--save-path", default="infer_result", type=str, metavar="PATH",
                        help="save results")

    args = parser.parse_args()

    # Build data loader
    test_set = TestDataset(args.data, (256, 256))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    net = model_mapper[args.arch]()
    Vtest = Predictor(net, test_loader, args.model,
                          args.spath, is_cuda)
    Vtest.predict()
