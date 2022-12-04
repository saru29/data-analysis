import React, { useState } from 'react'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'
import Webcam from './Webcam'
import JsonViewer from './JsonViewer'

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
  frameRate: { ideal: 30, max: 30 },
}

function CanvasWrapper() {
  const [camOn, setCamOn] = useState(false)
  const [pose, setPose] = useState({
    score: 'the model’s confidence on the identified pose',
    keypoints:
      'an array of key points – In this case, there are 33 key points and each key point contains the (x, y, z) coordinates, a confidence score for the keypoint and the name of the keypoint.',

    keypoints3D:
      'an array of 3D keypoints – each keypoint containing the name, confidence score and the (x, y, z) coordinates. These points represent the absolute distance in meters in a 2 x 2 x 2 meter cubic space. The range for each axis goes from -1 to 1 (therefore 2m total delta). The z is always perpendicular to the xy plane that passes the center of the hip, so the coordinate for the hip center is (0, 0, 0).',
  })

  return (
    <Grid container spacing={0} sx={{ justifyContent: 'space-around', mt: 5 }}>
      <Grid item={true}>
        <Webcam videoConstraints={videoConstraints} camOn={camOn} setPose={setPose} />
        <Grid
          item={true}
          md={12}
          sx={{ width: videoConstraints.width, height: videoConstraints.height }}
        >
          <Button variant="contained" size="small" sx={{ mt: 2 }} onClick={() => setCamOn(!camOn)}>
            webcam {camOn ? 'off' : 'on'}
          </Button>
        </Grid>
      </Grid>
      <Grid item={true}>
        <JsonViewer pose={pose} videoConstraints={videoConstraints} />
      </Grid>
    </Grid>
  )
}

export default CanvasWrapper
