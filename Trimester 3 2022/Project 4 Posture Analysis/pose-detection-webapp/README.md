# Pose Detection Web Application

## Overview

This is a ReactJS web application for pose detection, estimation and tracking using [BlazePose](https://blog.tensorflow.org/2021/05/high-fidelity-pose-tracking-with-mediapipe-blazepose-and-tfjs.html) pre-trained ML model.

The application enables the external webcam via a modern browser, and captures the human body movements in real time and displays the key points actively on a canvas that is on the top of the video stream.

The application also supports for sending the real time 3D key points info as JSON objects to a topic via MQTT protocol. The published key points data will be further utilised in the 3D cycling game to timely indicate a competitor cyclist's real pose movements.

## How to Use It Remotely

- This webapp has been deployed to the company owned GCP account at http://34.129.10.237:3003 (username: `admin`, password: `redback`).

- The app is exposed via a public static IP address only, which is not configured with HTTPS connection. Due to thee security restriction from modern browsers, to enable the webcam in a browser, however, a request connection has to be from HTTPS or localhost. To by-pass the limitation, please make sure you've added the remote IP address (`http://34.129.10.237:3003`) with the correct port number as a "secured origin connection" in your browser settings (e.g. `chrome://flags/#unsafely-treat-insecure-origin-as-secure` in Chrome).

## How to Run It Locally

1. In the root folder, run `npm install` to install all dependent packages.
2. Copy `.env.example` to `.env` and provide your own key values as needed.
3. Run `npm start` to start the app.
4. Go to `http://localhost:3003` in a browser to test out the features.

## How to Deploy it to GCP

- The webapp is containerized using Docker. A Docker config file for this app can be found in `.Dockerfile` file.
- For more details about how to build a new docker image, push to GCP, pull the new image in VM instances to deploy the latest app version, please refer to this deployment instruction guide: https://github.com/redbackoperations/iot/blob/main/docs/iot-web-services-deploy-guide.md    