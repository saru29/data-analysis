import React from 'react'
import { videoConstraints } from '../utils/videoHelper'

function Canvas({ camOn }) {
  return (
    camOn && (
      <canvas
        id="canvas"
        style={{
          position: 'absolute',
          display: 'block',
          zIndex: '100',
          width: videoConstraints.width,
          height: videoConstraints.height,
        }}
      ></canvas>
    )
  )
}

export default Canvas
