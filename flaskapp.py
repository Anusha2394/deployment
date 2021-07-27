# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:42:27 2020

@author: kasaa
"""

from flask import Flask,render_template
app = Flask(__name__)

posts = [
    {
     'author': 'Anusha',
     'title': 'Blog Post 1',
     'content': 'First Post Content',
     'date_posted': 'september 3 2020'
     },
    {
     'author': 'Anusha Kasa',
     'title': 'Blog Post 2',
     'content': 'second Post Content',
     'date_posted': 'september 3 2020'
     }
    ]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)

@app.route("/about")
def About():
    return render_template('about.html', title='About')

app.run(host="localhost", port=int("8080"))
