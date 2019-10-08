#!/usr/bin/env python
""" Web app for dosna over ceph to see and display objects"""
from flask import Flask, render_template
import argparse
import re
import numpy as np

import rados
import dosna as dn
import matplotlib.image as plt
import matplotlib
BACKEND = 'ceph'
ERROR = 'Object not found'


def parse_args():
    parser = argparse.ArgumentParser(description='Webapp')
    parser.add_argument('--connection-options', dest='connection_options',
                        nargs='+', default=[],
                        help='Cluster options using the format: '
                             'key1=val1 [key2=val2...]')
    return parser.parse_args()


app = Flask(__name__)


@app.route('/')
def index():
    """ Home page """
    return render_template('index.html')


@app.route('/list/obj/<pool>')
def list_object(pool):
    """ List objects in a pool """
    metaObj = ''  # The meta object
    mainObj = ''  # The actual object
    cluster = rados.Rados(**connection_config)
    cluster.connect()
    ioctx = cluster.open_ioctx(str(pool))
    objects = ioctx.list_objects()
    # Looping over objects
    for ob in objects:
        try:
            if (str(ob.key).__contains__('/')):
                mainObj += str(ob.key)+"<br>"
            else:
                metaObj += ("<b>Name: </b>"
                            + "<a href = /display/img/" + pool
                            + "/" + str(ob.key) + ">"
                            + str(ob.key) + "</a> <b>Contents: </b>"
                            + str(ob.read()) + "<b>Data type: </b>"
                            + str(ob.get_xattr('dtype')).replace('<', '')
                            + "<b> Shape:</b>"
                            + ob.get_xattr('shape') + "<br>")
        except (rados.Error,
                rados.IOError,
                rados.ObjectNotFound,
                rados.NoData,
                rados.NoSpace,
                rados.PermissionError,
                dn.backends.base.DatasetNotFoundError) as e:
            pass
    ioctx.close()
    cluster.shutdown()
    return render_template(
        'listObjects.html',
        pool=pool,
        metaObj=metaObj,
        mainObj=mainObj)


@app.route('/display/string/<pool>/<obj>')
def display_string_object(pool, obj):
    """ Display object as a string """
    np.set_printoptions(threshold=np.inf)
    cluster = rados.Rados(**connection_config)
    cluster.connect()
    ioctx = cluster.open_ioctx(str(pool))
    object_string = np.frombuffer(ioctx.read(str(obj)))
    object_shape = int(np.sqrt(object_string.size))
    object_string = object_string.reshape(object_shape, object_shape)
    object_string = str(object_string).replace('  ', ' ')
    object_string = object_string.replace('[', '<br>')
    object_string = object_string.replace(']', '<br>')
    ioctx.close()
    cluster.shutdown()
    return render_template(
        'objectString.html',
        obj=obj,
        object_string=object_string)


@app.route('/display/img/<pool>/<obj>')
def display_image_object(pool, obj):
    """ Display object as a image """
    fileFolder = 'static/'
    filename = str(obj) + '.jpeg'
    fileLocation = fileFolder + filename
    dn.use_backend(BACKEND)
    cluster = dn.Connection(str(pool), **connection_config)
    cluster.connect()
    try:
        object_data = cluster.get_dataset(str(obj))
        objectShape = object_data.instance.shape
        plt.imsave(
            fileLocation,
            object_data[:, (objectShape[1]/2), :],
            cmap=matplotlib.cm.gray)
        cluster.disconnect()
        return render_template('objectImage.html', filename=filename, obj=obj)
    except (rados.Error,
            rados.IOError,
            rados.ObjectNotFound,
            rados.NoData,
            rados.NoSpace,
            rados.PermissionError,
            dn.backends.base.DatasetNotFoundError) as e:
        cluster.disconnect()
        return render_template('objectImage.html', error=ERROR, obj=obj)


@app.route('/display/img/<pool>/<obj>/slice/<xslice>/<yslice>/<zslice>')
def display_image_object_slice(pool, obj, xslice, yslice, zslice):
    """ Display object as an image with slice specified """
    dn.use_backend(BACKEND)
    cluster = dn.Connection(str(pool), **connection_config)
    cluster.connect()
    try:
        object_data = cluster.get_dataset(str(obj))
        filename = makeImage(xslice, yslice, zslice, object_data, obj)
        cluster.disconnect()
        return render_template('objectImage.html', filename=filename, obj=obj)
    except (rados.Error,
            rados.IOError,
            rados.ObjectNotFound,
            rados.NoData,
            rados.NoSpace,
            rados.PermissionError,
            dn.backends.base.DatasetNotFoundError) as e:
        cluster.disconnect()
        return render_template('objectImage.html', error=ERROR, obj=obj)


def makeImage(xslice, yslice, zslice, object_data, obj):
    """ Converts object numpy array to an image """
    fileFolder = 'static/'
    if (':' in xslice and ':' in yslice):
        xslice = re.split(":", xslice)
        xslice = [int(i) for i in xslice]
        yslice = re.split(":", yslice)
        yslice = [int(i) for i in yslice]
        zslice = int(zslice)
        filename = (str(obj) + "#"
                    + str(xslice[0]) + ":" + str(xslice[1]) + "#"
                    + str(yslice[0]) + ":" + str(yslice[1]) + "#"
                    + str(zslice)
                    + '.png')
        fileLocation = fileFolder + filename
        plt.imsave(
            fileLocation,
            object_data[xslice[0]:xslice[1], yslice[0]:yslice[1], zslice],
            cmap=matplotlib.cm.gray)
    elif (':' in xslice and ':' in zslice):
        xslice = re.split(":", xslice)
        xslice = [int(i) for i in xslice]
        yslice = int(yslice)
        zslice = re.split(":", zslice)
        zslice = [int(i) for i in zslice]
        filename = (str(obj) + "#"
                    + str(xslice[0]) + ":" + str(xslice[1]) + "#"
                    + str(yslice) + "#"
                    + str(zslice[0]) + ":" + str(zslice[1])
                    + '.png')
        fileLocation = fileFolder + filename
        plt.imsave(
            fileLocation,
            object_data[xslice[0]:xslice[1], yslice, zslice[0]:zslice[1]],
            cmap=matplotlib.cm.gray)
    elif (':' in yslice and ':' in zslice):
        xslice = int(xslice)
        yslice = re.split(":", yslice)
        yslice = [int(i) for i in yslice]
        zslice = re.split(":", zslice)
        zslice = [int(i) for i in zslice]
        filename = (str(obj) + "#"
                    + str(xslice) + "#"
                    + str(yslice[0]) + ":" + str(yslice[1]) + "#"
                    + str(zslice[0]) + ":" + str(zslice[1])
                    + '.png')
        fileLocation = fileFolder + filename
        plt.imsave(
            fileLocation,
            object_data[xslice, yslice[0]:yslice[1], zslice[0]:zslice[1]],
            cmap=matplotlib.cm.gray)
    return filename


if __name__ == '__main__':
    args = parse_args()
    connection_config = {}
    connection_config.update(dict(
        item.split('=') for item in args.connection_options))
    app.run(debug=True, host='0.0.0.0')
