from madpack.datasets.base import DatasetBase
from madpack.datasets.cabc import CABC
from madpack.datasets.cifar10 import CIFAR10
from madpack.datasets.cub200 import CUB200_2011
from madpack.datasets.fss1000 import FSS1000
from madpack.datasets.imagenet import ILSVRC2012
from madpack.datasets.mnist import MNIST
from madpack.datasets.pathfinder import Pathfinder
from madpack.datasets.raven import Raven
from madpack.datasets.dtd import DescribableTextures
from madpack.datasets.square_count_dummy import SquareCountDummy
from madpack.datasets.cluttered_omniglot import ClutteredOmniglot, ClutteredOmniglotLoc, ClutteredOmniglotSeg, \
    ClutteredOmniglotXS
from madpack.datasets.rotating_objects import RotatingObjects


__all__ = ['DatasetBase', 'MNIST', 'CUB200_2011', 'ILSVRC2012', 'FSS1000', 'CABC', 'Pathfinder',
           'Raven', 'DescribableTextures', 'RotatingObjects']
