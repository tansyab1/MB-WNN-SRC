# Process CUSTOM data from .mat file
"""## 4. Process the **data**

### 4.3. Apply the Wavelet transform
    Arguments:
        image batch (sample, channel, width, height): batch of image to be transformed.
        level (int): level of wavelet transform.
"""

class MyDataset(Dataset):
    def __init__(self, mat_path, split_ratio, width, height):
        self.width = width
        self.height = height
        self.split_ratio = split_ratio
        feature = io.loadmat(mat_path, squeeze_me=True)['features']
        feature = feature[:, 0:2000]
        Label = io.loadmat(mat_path, squeeze_me=True)['Label']
        # print(Label.shape)
        Label = Label[0:2000]

        self.images = torch.from_numpy(np.transpose(feature)).type(torch.float)
        self.images = torch.reshape(self.images, [len(self.images), self.width, self.height])

        self.init_wt = torch.unsqueeze(self.images, dim=1)
        xfm = DWTForward(J=3, wave='haar')
        self.Yll, self.Yh = xfm(self.init_wt)
        self.Ylh, self.Yhl, self.Yhh = torch.unbind(self.Yh[0], dim=2)

        self.target = torch.from_numpy(np.transpose(Label)).type(torch.long)
        self.data = list(zip(self.Yll, self.Ylh, self.Yhl, self.Yhh, self.target))

        self.train_size = int(self.split_ratio * len(self.data))
        self.test_size = len(self.data) - self.train_size

    def _generate(self):
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.data,
                                                                              [self.train_size, self.test_size])
        return self.train_dataset, self.test_dataset

    def __getitem__(self, index):
        images = self.images[index]
        target = self.target[index]
        return images, target

    def __len__(self):
        return len(self.data)


class MySVHNDataset(datasets.SVHN):
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        super(MySVHNDataset, self).__init__(
            root, split, transform, target_transform, download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # Convert to grayscale
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target