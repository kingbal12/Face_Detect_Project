from flask import Flask, jsonify, escape, request
import cv2
import numpy as np
import urllib.request
import json
from wrinkle_analysis import measure_wrinkle, make_landmark_points
from landmark_rgb import find_rgb
from landmark_rgb_score import rgb_score

app = Flask(__name__)


@app.route("/")
def hello():
    name = request.args.get("name", "World")

    return f"Hello, {escape(name)}!"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
