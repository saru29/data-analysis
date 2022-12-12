import React, { useEffect, useTransition } from 'react'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Webcam from 'react-webcam'

import * as poseDetection from '@tensorflow-models/pose-detection'
import '@tensorflow/tfjs-backend-webgl'

import {
  videoConstraints,
  clearCanvas,
  drawKeypoints,
  drawSkeleton,
  sendKeypointsRate,
} from '../lib/videoHelper'

let detector, model, video, canvas, canvasContext
let frameCount = 0

const createDetector = async function () {
  model = poseDetection.SupportedModels.BlazePose
  const detectorConfig = {
    runtime: 'tfjs',
    enableSmoothing: true,
    modelType: 'full',
  }
  detector = await poseDetection.createDetector(model, detectorConfig)
}

function WebcamCapture({ camOn, setPose, publishPose }) {
  const [, startTransition] = useTransition()

  const initiateVideo = () => {
    video = document.getElementsByTagName('video')[0]
    canvas = document.getElementById('canvas')

    video.onloadedmetadata = () => {
      const videoWidth = video.videoWidth
      const videoHeight = video.videoHeight
      canvasContext = canvas.getContext('2d')

      canvas.width = videoWidth
      canvas.height = videoHeight
      canvasContext.translate(videoWidth, 0)
      canvasContext.scale(-1, 1)
    }

    // initially register the callback to be notified about the first frame.
    video.requestVideoFrameCallback(predictPoses)
  }

  const predictPoses = async function (now, metadata) {
    if (detector != null) {
      try {
        frameCount++
        const poses = await detector.estimatePoses(video, {
          flipHorizontal: false,
        })

        if (poses?.length > 0) {
          console.log('predicted poses data:', poses[0])

          clearCanvas()
          for (const pose of poses) {
            if (pose.keypoints !== null && typeof pose.keypoints == 'object') {
              // setPose(poses[0])
              drawSkeleton(poseDetection, model, canvasContext, pose.keypoints)
              drawKeypoints(canvasContext, pose.keypoints)
            }
          }

          if (frameCount >= sendKeypointsRate) {
            // start another transition to avoid for delaying the canvas re-draw
            startTransition(() => {
              publishPose(poses[0]?.keypoints3D || [])
              setPose(poses[0])
              frameCount = 0
            })
          }
        }
      } catch (error) {
        // detector.dispose()
        // detector = null
        console.log(error)
      }
    }

    // Re-register the callback for the next frame
    video.requestVideoFrameCallback(predictPoses)
  }

  useEffect(() => {
    if (!model) {
      createDetector()
        .then(() => console.log(`The ML model detector is`, detector))
        .catch(console.error)
    }
  }, [])

  return (
    <Paper
      elevation={3}
      sx={{
        width: videoConstraints.width,
        height: videoConstraints.height,
        display: 'inline-block',
      }}
    >
      {!camOn && (
        <Typography
          variant="h6"
          noWrap
          sx={{
            pt: '25%',
            fontFamily: 'monospace',
            fontWeight: 600,
            color: 'inherit',
            textDecoration: 'none',
          }}
        >
          Webcam streaming here...
        </Typography>
      )}

      {camOn && (
        <Webcam
          mirrored={true}
          audio={false}
          width={videoConstraints.width}
          height={videoConstraints.height}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          onUserMedia={initiateVideo}
        />
      )}
    </Paper>
  )
}

export default WebcamCapture
