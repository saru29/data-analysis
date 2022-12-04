import React from 'react'
import { JsonViewer } from '@textea/json-viewer'
import { videoConstraints } from '../utils/videoHelper'

function JsonViewerWrapper({ pose }) {
  return (
    <JsonViewer
      value={pose}
      theme="dark"
      style={{ width: '80vh', height: videoConstraints.height, overflow: 'auto' }}
    />
  )
}

export default JsonViewerWrapper
