import React, { useState } from 'react'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'

import Webcam from 'react-webcam'

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
}

function WebcamCapture() {
  const [camOn, setCamOn] = useState(false)

  return (
    <Grid container spacing={2}>
      <Grid md={12} display="flex" justifyContent="center">
        <Paper elevation={3} sx={{ width: 1280, height: 720 }}>
          {!camOn && (
            <Typography
              variant="h6"
              noWrap
              sx={{
                mt: '25%',
                fontFamily: 'monospace',
                fontWeight: 600,
                color: 'inherit',
                textDecoration: 'none',
              }}
            >
              Webcam streaming
            </Typography>
          )}

          {camOn && (
            <Webcam
              audio={false}
              height={720}
              screenshotFormat="image/jpeg"
              width={1280}
              videoConstraints={videoConstraints}
            >
              {/* {({ getScreenshot }) => (
              <button
                onClick={() => {
                  const imageSrc = getScreenshot()
                  console.log(imageSrc)
                }}
              >
                Capture photo
              </button>
            )} */}
            </Webcam>
          )}
        </Paper>
      </Grid>
      <Grid xs={12}>
        <Button variant="contained" size="small" onClick={() => setCamOn(!camOn)}>
          webcam {camOn ? 'off' : 'on'}
        </Button>
      </Grid>
    </Grid>
  )
}

export default WebcamCapture
