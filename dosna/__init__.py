

from .cluster import Cluster, Pool, File, ClusterException, connect, disconnect

from .dataset import Dataset, DatasetException

from .chunks import DataChunk, DataChunkException

from .utils import shape2str, str2shape, dtype2str


auto_init = connect