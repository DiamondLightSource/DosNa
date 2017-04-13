

from .cluster import Cluster, ClusterException, connect, disconnect

from .pool import Pool, File, PoolException

from .dataset import Dataset, DatasetException

from .chunk import DataChunk, DataChunkException

from .utils import shape2str, str2shape, dtype2str


auto_init = connect
