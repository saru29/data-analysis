import React from 'react'
import { JsonViewer } from '@textea/json-viewer'
import { videoConstraints } from '../lib/videoHelper'

function JsonViewerWrapper({ pose }) {
  return (
    <JsonViewer
      value={pose}
      theme="dark"
      style={{
        height: videoConstraints.height,
        overflow: 'auto',
      }}
    />
  )
}

export default JsonViewerWrapper
