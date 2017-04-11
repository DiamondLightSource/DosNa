

from .cluster import Cluster, Pool, File, ClusterException, connect, disconnect

from .dataset import DataChunk, Dataset, DatasetException

from .utils import shape2str, str2shape, dtype2str


auto_init = connect