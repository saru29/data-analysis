import React from 'react'
import { videoConstraints } from '../lib/videoHelper'

function Canvas({ camOn }) {
  return (
    camOn && (
      <canvas
        id="canvas"
        style={{
          position: 'absolute',
          display: 'inline-block',
          zIndex: '100',
          width: videoConstraints.width,
          height: videoConstraints.height,
        }}
      ></canvas>
    )
  )
}

export default Canvas
