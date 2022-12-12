import React, { useState } from 'react'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'

import Canvas from './Canvas'
import Webcam from './Webcam'
import JsonViewer from './JsonViewer'
import KeypointsSender from './KeypointsSender'
import { videoConstraints } from '../lib/videoHelper'
import mqttClient from './../lib/mqttClient'

let client

function CanvasWrapper() {
  const [camOn, setCamOn] = useState(false)
  const [pose, setPose] = useState({
    score: 'the model’s confidence on the identified pose',
    keypoints:
      'an array of key points – In this case, there are 33 key points and each key point contains the (x, y, z) coordinates, a confidence score for the keypoint and the name of the keypoint.',

    keypoints3D:
      'an array of 3D keypoints – each keypoint containing the name, confidence score and the (x, y, z) coordinates. These points represent the absolute distance in meters in a 2 x 2 x 2 meter cubic space. The range for each axis goes from -1 to 1 (therefore 2m total delta). The z is always perpendicular to the xy plane that passes the center of the hip, so the coordinate for the hip center is (0, 0, 0).',
  })

  const [bikeNumber, setBikeNumber] = useState('000001')
  const [mqttEnabled, setMqttEnabled] = useState(false)
  const [mqttStatus, setMqttStatus] = useState('not connected')
  const [sendingKeypointsInfo, setSendingKeypointsInfo] = useState('N/A')

  const enableMqtt = (enable) => {
    if (enable) {
      setMqttEnabled(true)
      console.log('connecting to MQTT ...')
      setMqttStatus('connecting...')
      client = mqttClient()

      client.on('connect', () => {
        console.log('Connected to the MQTT broker!')
        setMqttStatus('connected')
      })

      client.on('error', (error) => {
        console.log(error)
        setMqttStatus('failed to connect')
        setSendingKeypointsInfo('Cannot publish keypoints as the MQTT client is failed to connect!')
      })
    } else {
      setMqttEnabled(false)
      if (client) {
        console.log('disconnecting from MQTT ...')
        client.end()
        setMqttStatus('disconnected')
      }
    }
  }

  const publishPose = (keypoints) => {
    if (client && mqttStatus === 'connected') {
      client.publish(
        process.env.REACT_APP_BIKE_POSE_KEYPOINTS_MQTT_TOPIC_REGEX.replace('+', bikeNumber),
        JSON.stringify(keypoints)
      )
      setSendingKeypointsInfo(
        `[${new Date().toISOString()}] A new keypoints set has just been sent!`
      )
    } else {
      setSendingKeypointsInfo('Cannot publish keypoints as the MQTT client is not connected yet!')
    }
  }

  return (
    <Grid container spacing={0} sx={{ justifyContent: 'space-between', mt: 5 }}>
      <Grid item={true} md={8} xs={12}>
        <Canvas camOn={camOn} />
        <Webcam camOn={camOn} setPose={setPose} publishPose={publishPose} />
        <Grid item={true} sx={{ width: videoConstraints.width, display: 'inline-block' }}>
          <Button
            variant="contained"
            size="small"
            sx={{ mt: 2, mb: 2 }}
            onClick={() => {
              setCamOn(!camOn)

              setSendingKeypointsInfo('N/A')
            }}
          >
            webcam {camOn ? 'off' : 'on'}
          </Button>
        </Grid>
      </Grid>
      <Grid item={true} md={4} xs={12}>
        <JsonViewer pose={pose} />
        <Grid item={true} md={12}>
          <KeypointsSender
            camOn={camOn}
            enableMqtt={enableMqtt}
            bikeNumber={bikeNumber}
            mqttEnabled={mqttEnabled}
            mqttStatus={mqttStatus}
            sendingKeypointsInfo={sendingKeypointsInfo}
            setBikeNumber={setBikeNumber}
          />
        </Grid>
      </Grid>
    </Grid>
  )
}

export default CanvasWrapper
