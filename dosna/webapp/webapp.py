#!/usr/bin/env python
""" Web app for dosna over ceph to see and display objects"""
from flask import Flask, render_template
import argparse
import numpy as np

import rados
import dosna as dn

from PIL import Image  

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
    # Looping over objects
    for ob in objects:
        try:	
            if (str(ob.key).__contains__('*')):
                mainObjData += "<a href= /display/string/"+pool+"/"+ str(ob.key)+ ">"+ str(ob.key)+ "</a> <br>"	
            else:
                headObjData += "<b>Name: </b>"+ str(ob.key) +  " <b>Contents: </b>" + str(ob.read()) +" <b>Data type: </b>" +  str(ob.get_xattr('dtype')).replace('<','') + " <b> Shape:</b>"+ ob.get_xattr('shape') + "<br>"
        except:
            pass	
    ioctx.close()
    cluster.shutdown()
    return render_template('listObjects.html',pool=pool,metaObj=metaObj,mainObj=mainObj)

# Display object as a string
@app.route('/display/string/<pool>/<obj>')
def display_string_object(pool,obj):
    return render_template('objectString.html',obj=obj)

# Display object as a image
@app.route('/display/img/<pool>/<obj>')
def display_image_object(pool,obj):
    return render_template('objectImage.html',obj=obj)

if __name__ == '__main__':
    args = parse_args()
	connection_config = {}
	connection_config.update(dict(item.split('=') for item in args.connection_options))
	app.run(debug=True, host='0.0.0.0')