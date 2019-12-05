from flask import Flask, render_template, url_for, request
app = Flask(__name__)

import numpy as np

import matplotlib
import json
import random

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from threading import Lock
lock = Lock()
import datetime
import mpld3
from mpld3 import plugins

from visualization import visualizer

def draw_fig(fig_type):
    """Returns html equivalent of matplotlib figure
    Parameters
    ----------
    fig_type: string, type of figure
            one of following:
                    * line
                    * bar
    Returns
    --------
    d3 representation of figure
    """

    with lock:
        fig, ax = plt.subplots()
        if fig_type == "line":
            ax.plot(x, y)
        elif fig_type == "bar":
            ax.bar(x, y)
        elif fig_type == "pie":
            ax.pie(pie_fracs, labels=pie_labels)
        elif fig_type == "scatter":
            ax.scatter(x, y)
        elif fig_type == "hist":
            ax.hist(y, 10, normed=1)
        elif fig_type == "area":
            ax.plot(x, y)
            ax.fill_between(x, 0, y, alpha=0.2)


    return mpld3.fig_to_html(fig)

@app.route('/')
@app.route('/Home')
@app.route('/Home', methods=['POST'])
def home():
    select = request.form.get('co-select')
    select_text = ""
    ret = 'Select Data'
    vis = visualizer()
    if select == 'fbf':
        ret = vis.visualizeFbFeed()
    elif select == 'fba':
        ret = vis.visualizeFbAll()
    elif select == 'apf':
        ret = vis.visualizeApFeed()
    elif select == 'apa':
        ret = vis.visualizeApAll()
    elif select == 'amf':
        ret = vis.visualizeAmFeed()
    elif select == 'ama':
        ret = vis.visualizeAmAll()
    elif select == 'nef':
        ret = vis.visualizeNfFeed()
    elif select == 'nea':
        ret = vis.visualizeNfAll()
    elif select == 'gof':
        ret = vis.visualizeGoFeed()
    elif select == 'goa':
        ret = vis.visualizeGoAll()
    return render_template('home.html', title=select_text, plot=ret)


@app.route('/Info')
def info():
    return render_template('info.html', title='info')


if __name__ == '__main__':
    app.run(debug=True)
