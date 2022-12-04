import React from 'react'

import { JsonViewer } from '@textea/json-viewer'

function JsonViewerWrapper({ videoConstraints, pose }) {
  return (
    <JsonViewer
      value={pose}
      theme="dark"
      style={{ width: '80vh', height: videoConstraints.height, overflow: 'auto' }}
    />
  )
}

export default JsonViewerWrapper
