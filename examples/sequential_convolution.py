#!/usr/bin/env python
"""3D convolution using a gaussian filter

All the data is managed using dosna"""

from __future__ import print_function

import argparse
import json
from os.path import join
from time import strftime

import numpy as np

try:
    from imageio import imwrite
except:
    pass

from scipy.ndimage import gaussian_filter, gaussian_filter1d

import dosna as dn
from dosna.util import Timer

PREFIX = strftime("%y%m%d%H%M%S")
IN_DS_NAME = '{}_data_'.format(PREFIX)
DS1_NAME = '{}_gaussian_3d'.format(IN_DS_NAME)
DS2X_NAME = '{}_gaussian_x'.format(IN_DS_NAME)
DS2Y_NAME = '{}_gaussian_y'.format(IN_DS_NAME)
DS2Z_NAME = '{}_gaussian_z'.format(IN_DS_NAME)
RESULT = "{}_result.json".format(PREFIX)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Gaussian Convolution')
    parser.add_argument('--backend', dest='backend', default='hdf5',
                        help='Select backend to use (ram | *hdf5 | ceph)')
    parser.add_argument('--engine', dest='engine', default='mpi',
                        help='Select engine to use (cpu | joblib | *mpi)')
    parser.add_argument('--connection', dest='connection', default='test-dosna',
                        help='Connection name')
    parser.add_argument('--connection-options', dest='connection_options',
                        nargs='+', default=[],
                        help='Cluster options using the format: '
                             'key1=val1 [key2=val2...]')
    parser.add_argument('--out', dest='out', default='.',
                        help='Output directory for the results (default ".").')

    parser.add_argument('--data-sizes', dest='data_sizes', type=int, nargs='+',
                        default=[128, 256, 512],
                        help='List of sizes for datasets to test in. '
                        'Sizes are 3D, e.g. 128 -> 128x128x128')
    parser.add_argument('--chunk-sizes', dest='chunk_sizes', type=int, nargs='+',
                        default=[24, 36, 48, 60, 72, 96, 128],
                        help='List of sizes for chunking the datasets. '
                        'Sizes are 3D, e.g. 32 -> 32x32x32')

    parser.add_argument('--sigma', dest='sigma', type=float, default=1.5,
                        help='Determines the sigma to be used for the '
                        'Gaussian Convolution')

    parser.add_argument('--ntest', dest='ntest', type=int, default=10,
                        help='Number of tests to be run for each data size '
                        'and chunk size')

    return parser.parse_args()


def create_random_dataset(DS):
    return np.random.rand(DS, DS, DS).astype(np.float32)


def convolve1(ds, sigma, out_path):

    print('Convolving in 3D')

    ds3d_ = ds.clone(DS1_NAME)
    timer = Timer('Separable 3D convolution')

    with timer:
        for z in range(ds.chunk_grid[0]):
            zS = z * ds.chunk_size[0]
            zE = min(zS + ds.chunk_size[0], ds.shape[0])
            for y in range(ds.chunk_grid[1]):
                yS = y * ds.chunk_size[1]
                yE = min(yS + ds.chunk_size[1], ds.shape[1])
                for x in range(ds.chunk_grid[2]):
                    xS = x * ds.chunk_size[2]
                    xE = min(xS + ds.chunk_size[2], ds.shape[2])
                    out = gaussian_filter(ds[zS:zE, yS:yE, xS:xE], sigma=sigma)
                    ds3d_[zS:zE, yS:yE, xS:xE] = out

    try:
        imwrite(join(out_path,
                     '{}_conv3d_{}-{}.png'.format(PREFIX, ds.shape[0],
                                                  ds.chunk_size[0])),
                (ds3d_[ds.shape[0] // 2]*255).astype(np.uint8))
    except NameError:
        pass
    except Exception as ex:
        print('Skipping image write due to errors: {}'.format(
            ex.message))

    ds3d_.delete()

    return timer.time


def convolve2(ds, sigma, out_path):

    timer = Timer('Three separable 1D convolutions')

    with timer:
        print('Convolving axis Z')

        ds1_ = ds.clone(DS2Z_NAME)

        for z in range(ds.chunk_grid[0]):
            zS = z * ds.chunk_size[0]
            zE = min(zS + ds.chunk_size[0], ds.shape[0])
            out = gaussian_filter1d(ds[zS:zE], sigma=sigma, axis=0)
            ds1_[zS:zE] = out

        # Gaussian second axis

        print('Convolving axis Y')

        ds2_ = ds.clone(DS2Y_NAME)

        for y in range(ds.chunk_grid[1]):
            yS = y * ds.chunk_size[1]
            yE = min(yS + ds.chunk_size[1], ds.shape[1])
            out = gaussian_filter1d(ds1_[:, yS:yE], sigma=sigma, axis=1)
            ds2_[:, yS:yE] = out

        # Gaussian second axis

        print('Convolving axis X')

        ds3_ = ds.clone(DS2X_NAME)

        for x in range(ds.chunk_grid[2]):
            xS = x * ds.chunk_size[2]
            xE = min(xS + ds.chunk_size[2], ds.shape[2])
            out = gaussian_filter1d(ds2_[..., xS:xE], sigma=sigma, axis=2)
            ds3_[..., xS:xE] = out

    try:
        imwrite(join(out_path, '{}_conv3x1d_{}-{}.png'.format(PREFIX,
                                                              ds.shape[0],
                                                              ds.chunk_size[0])),
                (ds3_[ds.shape[0] // 2]*255).astype(np.uint8))
    except NameError:
        pass
    except Exception as ex:
        print('Skipping image write due to errors: {}'.format(
            ex.message))

    ds1_.delete()
    ds2_.delete()
    ds3_.delete()

    return timer.time


def main():
    args = parse_args()
    data_sizes = args.data_sizes
    chunk_sizes = args.chunk_sizes

    dn.use(backend=args.backend, engine=args.engine)

    ntest = args.ntest
    sigma = args.sigma
    connection_config = {"name": args.connection}
    connection_config.update(
        dict(item.split('=') for item in args.connection_options))
    out_path = args.out
    result_info = []

    engine, backend = dn.status()
    print('Starting Test == Backend: {}, Engine: {}, config: {}, Out: {}'
          .format(backend.name, engine.name, connection_config, out_path))
    connection = dn.Connection(**connection_config)
    connection.connect()
    for i, DS in enumerate(data_sizes):

        data = create_random_dataset(DS)

        for j, CS in enumerate(chunk_sizes):
            if CS > DS:
                print('Skipping not valid parameters'
                      ' -- shape: {} chunk_size: {}'.format(DS, CS))
                continue

            print('Loading Data -- shape: {} chunk_size: {}'.format(DS, CS))
            t = Timer('Data loaded')
            with t:
                dataset = connection.create_dataset(IN_DS_NAME, data=data,
                                                    chunk_size=(CS, CS, CS))

            result_info.append({
                'create_time': t.time,
                'conv1_times': [],
                'conv2_times': [],
                'dataset_size': DS,
                'datachunk_size': CS,
                'engine': args.engine,
                'backend': args.backend
            })

            for k in range(ntest):
                t1 = convolve1(dataset, sigma, out_path)
                result_info[-1]['conv1_times'].append(t1)
                t2 = convolve2(dataset, sigma, out_path)
                result_info[-1]['conv2_times'].append(t2)
            t = Timer('Data deleted')
            with t:
                dataset.delete()
            result_info[-1]['delete_time'] = t.time
            # write after every test, to not loose a lot of data when
            # there are errors
            json.dump(result_info, open(join(out_path, RESULT), "w"),
                      sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()
