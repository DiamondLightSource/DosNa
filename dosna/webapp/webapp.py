#!/usr/bin/env python
""" Web app for dosna over ceph to see and display objects"""
from flask import Flask, render_template

import numpy as np

import rados
import dosna as dn

from PIL import Image  

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Listing objects inside pool
@app.route('/list/obj/<pool>')
def list_object(pool):
    return render_template('listObjects.html')

# Display object as a string
@app.route('/display/string/<pool>/<obj>')
def display_string_object(pool,obj):
    return render_template('objectString.html',obj=obj)

# Display object as a image
@app.route('/display/img/<pool>/<obj>')
def display_image_object(pool,obj):
    return render_template('objectImage.html',obj=obj)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')