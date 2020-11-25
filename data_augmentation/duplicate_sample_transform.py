class DuplicateSampleTransform(object):
    def __init__(self, transform):
        """
        This class is designed to apply the same data transformation given twice to the same sample and integrate
        like any tochvision.transforms
        :param transform: list of torchvision.transforms
        """
        self.transform = transform

    def __call__(self, sample):
        """
        Will  apply the given list of transforms twice on the same initial sample and return back transformed tensors
        :param sample: torch.DataLoader item
        :return: (torch.Tensor, torch.Tensor)
        """
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj
