#!/usr/bin/env python
""" Web app for dosna over ceph to see and display objects"""
from flask import Flask, render_template
import argparse
import re
import numpy as np

import rados
import dosna as dn

from PIL import Image  
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
# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Listing objects inside pool
@app.route('/list/obj/<pool>')
def list_object(pool):
    metaObj= '' # The meta object
    mainObj= '' # The actual object
    cluster = rados.Rados(**connection_config)
    cluster.connect()
    ioctx = cluster.open_ioctx(str(pool))
    objects = ioctx.list_objects()
    # Looping over objects
    for ob in objects:
        try:	
            if (str(ob.key).__contains__('/')):
                mainObj += "<a href= /display/string/"+pool+"/"+ str(ob.key)+ ">"+ str(ob.key)+ "</a> <br>"	
            else:
                metaObj += "<b>Name: </b>"+ "<a href = /display/img/"+pool+"/"+str(ob.key) + ">" +str(ob.key) + "</a> <b>Contents: </b>" + str(ob.read()) +" <b>Data type: </b>" +  str(ob.get_xattr('dtype')).replace('<','') + " <b> Shape:</b>"+ ob.get_xattr('shape') + "<br>"
        except:
            pass	
    ioctx.close()
    cluster.shutdown()
    return render_template('listObjects.html',pool=pool,metaObj=metaObj,mainObj=mainObj)

# Display object as a string
@app.route('/display/string/<pool>/<obj>')
def display_string_object(pool,obj):
    np.set_printoptions(threshold=np.inf)
    cluster = rados.Rados(**connection_config)
    cluster.connect()
    ioctx = cluster.open_ioctx(str(pool))
    object_string = np.frombuffer(ioctx.read(str(obj)))
    object_shape = int(np.sqrt(object_string.size))
    object_string = object_string.reshape(object_shape,object_shape)
    object_string = str(object_string).replace('  ',' ')
    object_string=object_string.replace('[','<br>')
    object_string=object_string.replace(']','<br>')
    ioctx.close()
    cluster.shutdown()
    return render_template('objectString.html',obj=obj,object_string=object_string)

# Display object as a image
@app.route('/display/img/<pool>/<obj>')
def display_image_object(pool,obj):
    dn.use_backend(BACKEND)
    cluster = dn.Connection(str(pool),**connection_config)
    cluster.connect()
    try:
        object_data = cluster.get_dataset(str(obj))
        img = Image.fromarray(object_data[:,:,0],'RGB')  
        cluster.disconnect()
        fileLocation = 'static/' +str(obj)+'.jpeg'
        filename = obj+'.jpeg'
        img.save(fileLocation)
        return render_template('objectImage.html',filename=filename,obj=obj)
    except:
        cluster.disconnect()
        return render_template('objectImage.html',error=ERROR,obj=obj)

if __name__ == '__main__':
    args = parse_args()
    connection_config = {}
    connection_config.update(dict(item.split('=') for item in args.connection_options))
    app.run(debug=True, host='0.0.0.0')