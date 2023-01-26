from interface import (load_config, load_algo, load_data, load_model, load_utils)
import torch, random, numpy
from torchvision.utils import make_grid
from torchvision import transforms as T
import numpy as np
from PIL import Image

class Scheduler():
    def __init__(self) -> None:
        pass

    def assign_config_by_path(self, config_path) -> None:
        self.config = load_config(config_path)

    def assign_config_by_dict(self, config) -> None:
        self.config = config

    def initialize(self) -> None:
        assert self.config is not None, "Config should be set when initializing"
        self.utils = load_utils(self.config)

        # set seeds
        seed = self.config["seed"]
        torch.manual_seed(seed);
        random.seed(seed);
        numpy.random.seed(seed)

        if not self.config["experiment_type"] == "challenge":
            self.utils.copy_source_code(self.config["results_path"])
        self.utils.init_logger()

        self.dataloader = load_data(self.config)

        self.algo = load_algo(self.config, self.utils, self.dataloader.train)

        if self.config["experiment_type"] == "defense":
            self.model = load_model(self.config, self.utils)

        self.utils.logger.set_log_freq(len(self.dataloader.train))

    def run_job(self):
        self.utils.logger.log_console("Starting the job")
        exp_type = self.config["experiment_type"]
        if exp_type == "challenge":
            self.run_challenge_job()
        elif exp_type == "defense":
            self.run_defense_job()
        elif exp_type == "attack":
            self.run_attack_job()
        else:
            print("unknown experiment type")

    def run_defense_job(self):
        for epoch in range(self.config["total_epochs"]):
            self.defense_train(epoch)
            self.defense_test(epoch)
            self.epoch_summary()
        self.generate_challenge()

    def run_attack_job(self):
        print("running attack job")
        for epoch in range(self.config["total_epochs"]):
            self.attack_train()
            self.attack_test()
            self.epoch_summary()

    def run_challenge_job(self):
        self.utils.load_saved_models()
        self.generate_challenge()

    def defense_train(self, epoch) -> None:
        self.algo.train()
        self.model.train()
        trans = T.ToPILImage()
        for ind, sample in enumerate(self.dataloader.train):
            items = self.utils.get_data(sample)
            z, _ = self.algo.forward(items)
            if ind == 0:
                z, x_recon = self.algo.forward(items)

                input_alt = trans(items["x"][0])
                input_alt.save(f"samples_18_3_50/real_train_alt_{epoch}.png")
                recon_alt = trans(x_recon[0])
                recon_alt.save(f"samples_18_3_50/recon_train_alt_{epoch}.png")

                # recon_np = x_recon.cpu().detach().numpy()
                # input_np = items["x"].cpu().detach().numpy()
                # first_image = recon_np[0]
                # first_image = first_image.transpose(1, 2, 0)
                # first_image_orig = input_np[0]
                # first_image_orig = first_image_orig.transpose(1, 2, 0)
                # Image.fromarray(first_image, 'RGB').save(f"samples/recons_train_{epoch}.png")
                # Image.fromarray(first_image_orig, 'RGB').save(f"samples/real_train_{epoch}.png")

            #             print(f"shape of z before server model {z.shape}")
            else:
                z, _ = self.algo.forward(items)
            data = self.model.forward(z)
            items["decoder_grads"] = self.algo.infer(data, items["pred_lbls"])
            items["server_grads"] = self.model.backward(items["pred_lbls"], items["decoder_grads"])
            self.algo.backward(items)
            #             print(f"shape of z before

    def defense_test(self, epoch) -> None:
        self.algo.eval()
        self.model.eval()
        trans = T.ToPILImage()
        for ind, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            if ind == 0:
                z, x_recon = self.algo.forward(items)

                input_alt = trans(items["x"][0])
                input_alt.save(f"samples_18_3_50/real_test_alt_{epoch}.png")
                recon_alt = trans(x_recon[0])
                recon_alt.save(f"samples_18_3_50/recon_test_alt_{epoch}.png")

                # recon_np = x_recon.cpu().detach().numpy()
                # input_np = items["x"].cpu().detach().numpy()
                # first_image = recon_np[0]
                # first_image = first_image.transpose(1, 2, 0)
                # first_image_orig = input_np[0]
                # first_image_orig = first_image_orig.transpose(1, 2, 0)
                # Image.fromarray(first_image, 'RGB').save(f"samples/recons_test_{epoch}.png")
                # Image.fromarray(first_image_orig, 'RGB').save(f"samples/real_test_{epoch}.png")

            else:
                z, _ = self.algo.forward(items)

            data = self.model.forward(z)
            self.algo.infer(data, items["pred_lbls"])
            self.model.compute_loss(data, items["pred_lbls"])

    def attack_train(self) -> None:
        if self.config.get("no_train"):
            return
        self.algo.train()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.algo.backward(items)

    def attack_test(self):
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.utils.save_images(z, items["filename"])

    def epoch_summary(self):
        self.utils.logger.flush_epoch()
        self.utils.save_models()

    def generate_challenge(self) -> None:
        challenge_dir = self.utils.make_challenge_dir(self.config["results_path"])
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.utils.save_data(z, items["filename"], challenge_dir)
