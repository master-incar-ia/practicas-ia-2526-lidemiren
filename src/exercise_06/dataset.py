from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import scipy
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(
        self,
        ecg_file="./data/ecg/cut_pub_4sec.mat",
        train=True,
    ):
        self.data = self._import_data(ecg_file)
        train_test_subset = numpy.array(self.data["train"])
        X = numpy.array(self.data["s_ecg"])
        Y = numpy.array(self.data["s_ns"])
        # convert Y to one-hot encoding
        Y = numpy.eye(2)[Y]
        self.classes = ["shockable", "not_shockable"]

        if train:
            self.X = X[train_test_subset == 1]
            self.Y = Y[train_test_subset == 1]
        else:
            self.X = X[train_test_subset == 0]
            self.Y = Y[train_test_subset == 0]

    def _import_data(self, file_name):
        # Load the data, expand data, and provide train/test set (no CV yet)

        data = scipy.io.loadmat(file_name)["data"][0]

        num_samples = len(data["name"])
        pat_ID = []
        s_ns = []
        name = []
        r_type = []
        s_ecg = []
        source = []
        train = []

        for i in range(num_samples):
            try:
                pat_ID.append(data["pat_ID"][i][0][0])
            except:
                pat_ID.append(-1)

            name.append(data["name"][i][0])
            s_ns.append(data["s_ns"][i][0][0])
            r_type.append(data["r_type"][i][0])
            s_ecg.append(data["s_ecg"][i][0, :])
            try:
                source.append(data["source"][i][0][0])
            except:
                source.append(data["o_ddbb"][i][0][0])

            try:
                train.append(data["train"][i][0][0])
            except:
                train.append(-1)

        output_data = {
            "pat_ID": pat_ID,
            "s_ns": s_ns,
            "name": name,
            "r_type": r_type,
            "s_ecg": s_ecg,
            "source": source,
            "train": train,
        }
        return output_data

    def _extend_data(self, data, data_interval_seconds=5.0, step_seconds=1.0):
        samples_per_second = 250
        data_interval = int(data_interval_seconds * samples_per_second)
        step = int(step_seconds * samples_per_second)
        len_signal = data["s_ecg"][0].shape[0]

        pat_ID = []
        s_ns = []
        name = []
        r_type = []
        s_ecg = []
        source = []
        pat_ID = []
        train = []

        for element in range(len(data["name"])):
            for init_index in range(0, len_signal - data_interval, step):
                pat_ID.append(data["pat_ID"][element])
                s_ns.append(data["s_ns"][element])
                name.append(data["name"][element] + "_%d" % init_index)
                r_type.append(data["r_type"][element])
                source.append(data["source"][element])
                train.append(data["train"][element])
                s_ecg.append(data["s_ecg"][element][init_index : (init_index + data_interval)])

        output_data = {
            "pat_ID": pat_ID,
            "s_ns": s_ns,
            "name": name,
            "r_type": r_type,
            "s_ecg": s_ecg,
            "source": source,
            "train": train,
        }
        return output_data

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def plot(self, filepath):
        plt.figure(figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                x = self.X[i * 10 + j]

                if self.Y[i * 10 + j][1]:
                    plt.plot(x, "r")
                else:
                    plt.plot(x, "b")
                plt.xlabel(self.classes[int(self.Y[i * 10 + j][1])])
        plt.show()
        plt.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name 
    output_folder.mkdir(exist_ok=True, parents=True)

    # Include extend data!!!
    dataset_train = ECGDataset(train=True)
    dataset_test = ECGDataset(train=False)
    print(f"Dataset length: {len(dataset_train)}")
    print(f"First item: {dataset_train[0]}")
    dataset_train.plot(output_folder / "plot_dataset_example.png")
