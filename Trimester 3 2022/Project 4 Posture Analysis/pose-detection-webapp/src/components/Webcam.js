import React, { useEffect } from 'react'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Webcam from 'react-webcam'

import * as poseDetection from '@tensorflow-models/pose-detection'
import '@tensorflow/tfjs-backend-webgl'

let detector, model, video

const createDetector = async function () {
  model = poseDetection.SupportedModels.BlazePose
  const detectorConfig = {
    runtime: 'tfjs',
    enableSmoothing: true,
    modelType: 'full',
  }
  detector = await poseDetection.createDetector(model, detectorConfig)
}

function WebcamCapture({ videoConstraints, camOn, setPose }) {
  const initiateVideo = () => {
    video = document.getElementsByTagName('video')[0]

    video.onloadedmetadata = () => {
      const videoWidth = video.videoWidth
      const videoHeight = video.videoHeight

      video.width = videoWidth
      video.height = videoHeight
      // canvas.width = videoWidth
      // canvas.height = videoHeight

      // Because the image from camera is mirrored, need to flip horizontally.
      // ctx.translate(videoWidth, 0)
      // ctx.scale(-1, 1)
    }

    // Initially register the callback to be notified about the first frame.
    video.requestVideoFrameCallback(predictPoses)
  }

  const predictPoses = async function (now, metadata) {
    if (detector != null) {
      try {
        const poses = await detector.estimatePoses(video, {
          flipHorizontal: false,
        })

        if (poses?.length > 0) {
          console.log('predicted poses data:', poses[0])
          setPose(poses[0])
        }
      } catch (error) {
        // detector.dispose()
        // detector = null
        console.log(error)
      }
    }
    // ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight)

    // if (poses && poses.length > 0) {
    //   for (const pose of poses) {
    //     if (pose.keypoints != null) {
    //       drawKeypoints(pose.keypoints)
    //       drawSkeleton(pose.keypoints)
    //     }
    //   }
    // }

    console.log(`[${now}]: current video frame data is`, metadata)

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
    <Paper elevation={3} sx={{ width: videoConstraints.width, height: videoConstraints.height }}>
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
