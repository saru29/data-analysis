const scoreThreshold = 0.6

const videoConstraints = {
  width: 800,
  height: 500,
  facingMode: 'user',
  frameRate: { ideal: 30, max: 30 },
}

// send a new Pose data to MQTT every 30/2 frames, for instance
const sendKeypointsRate = videoConstraints.frameRate.max / 2

const clearCanvas = () => {
  const canvas = document.getElementById('canvas')

  if (canvas) {
    const canvasContext = canvas.getContext('2d')
    canvasContext.clearRect(0, 0, canvas.width, canvas.height)
  }
}

const drawKeypoints = (canvasContext, keypoints) => {
  canvasContext.fillStyle = 'Blue'
  canvasContext.strokeStyle = 'White'
  canvasContext.lineWidth = 2

  for (let i = 0; i < keypoints.length; i++) {
    drawKeypoint(canvasContext, keypoints[i])
  }
}

const drawKeypoint = (canvasContext, keypoint) => {
  const radius = 4
  if (keypoint.score >= scoreThreshold) {
    const circle = new Path2D()
    circle.arc(keypoint.x, keypoint.y, radius, 0, 2 * Math.PI)
    canvasContext.fill(circle)
    canvasContext.stroke(circle)
  }
}

const drawSkeleton = (poseDetection, model, canvasContext, keypoints) => {
  const color = '#fff'
  canvasContext.fillStyle = color
  canvasContext.strokeStyle = color
  canvasContext.lineWidth = 2

  poseDetection.util.getAdjacentPairs(model).forEach(([i, j]) => {
    const kp1 = keypoints[i]
    const kp2 = keypoints[j]
    if (kp1.score >= scoreThreshold && kp2.score >= scoreThreshold) {
      canvasContext.beginPath()
      canvasContext.moveTo(kp1.x, kp1.y)
      canvasContext.lineTo(kp2.x, kp2.y)
      canvasContext.stroke()
    }
  })
}

export {
  videoConstraints,
  scoreThreshold,
  clearCanvas,
  drawKeypoints,
  drawSkeleton,
  sendKeypointsRate,
}
