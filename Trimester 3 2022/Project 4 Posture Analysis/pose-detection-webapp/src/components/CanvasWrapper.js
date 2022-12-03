import * as React from 'react'
import Box from '@mui/material/Box'

import Webcam from './Webcam'

function CanvasWrapper() {
  return (
    <Box
      sx={{
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        mt: 3,
        '& > :not(style)': {
          m: 1,
          width: '100%',
          minHeight: 800,
        },
      }}
    >
      <Webcam />
    </Box>
  )
}

export default CanvasWrapper
