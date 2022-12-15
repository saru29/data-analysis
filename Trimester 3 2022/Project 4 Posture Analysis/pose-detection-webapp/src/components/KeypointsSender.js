import React from 'react'
import Chip from '@mui/material/Chip'
import Typography from '@mui/material/Typography'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import Switch from '@mui/material/Switch'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem'
import FormControl from '@mui/material/FormControl'
import Select from '@mui/material/Select'

function KeypointsSender({
  camOn,
  enableMqtt,
  bikeNumber,
  mqttEnabled,
  mqttStatus,
  sendingKeypointsInfo,
  setBikeNumber,
  MsgCount,
}) {
  const handleBikeNumberChange = (event) => {
    setBikeNumber(event.target.value)
  }

  return (
    <>
      <List dense={false}>
        <ListItem sx={{ justifyContent: 'center' }}>
          <Typography
            variant="p"
            sx={{
              fontFamily: 'monospace',
              fontWeight: 400,
              color: 'inherit',
              textDecoration: 'none',
              mr: 1,
            }}
          >
            Enable MQTT:
          </Typography>
          <Switch
            inputProps={{ 'aria-label': 'Switch demo' }}
            color="secondary"
            onClick={() => enableMqtt(!mqttEnabled)}
            disabled={camOn}
          />
        </ListItem>
        <Typography
          variant="p"
          sx={{
            fontFamily: 'monospace',
            fontWeight: 400,
            color: 'inherit',
            textDecoration: 'none',
            mr: 1,
          }}
        >
          (please turn off the camera before enabling MQTT)
        </Typography>
        <ListItem sx={{ justifyContent: 'center' }}>
          <Typography
            variant="p"
            sx={{
              fontFamily: 'monospace',
              fontWeight: 400,
              color: 'inherit',
              textDecoration: 'none',
              mr: 1,
            }}
          >
            MQTT status:
          </Typography>
          <Chip
            label={mqttStatus}
            style={{ fontWeight: '500px' }}
            color={
              mqttStatus === 'not connected' || mqttStatus === 'disconnected'
                ? 'warning'
                : mqttStatus === 'connected'
                ? 'success'
                : 'default'
            }
          />
        </ListItem>
        <ListItem sx={{ justifyContent: 'center' }}>
          <Typography
            variant="p"
            sx={{
              fontFamily: 'monospace',
              fontWeight: 400,
              color: 'inherit',
              textDecoration: 'none',
              mr: 1,
            }}
          >
            Bike Number:
          </Typography>
          <FormControl>
            <InputLabel id="demo-simple-select-label">bike number</InputLabel>
            <Select
              labelId="select-bike-number-label"
              value={bikeNumber}
              label="BikeNumber"
              onChange={handleBikeNumberChange}
              sx={{ width: 'fit-content' }}
            >
              <MenuItem value={'000001'}>000001</MenuItem>
              <MenuItem value={'000002'}>000002</MenuItem>
            </Select>
          </FormControl>
        </ListItem>

        <ListItem sx={{ justifyContent: 'center' }}>
          <Typography
            variant="p"
            sx={{
              fontFamily: 'monospace',
              fontWeight: 400,
              color: 'inherit',
              textDecoration: 'none',
              mr: 1,
            }}
          >
            Data Sending Topic:
          </Typography>
          <Chip
            label={process.env.REACT_APP_BIKE_POSE_KEYPOINTS_MQTT_TOPIC_REGEX.replace(
              '+',
              bikeNumber
            )}
            style={{ fontWeight: '500px' }}
          />
        </ListItem>
        <ListItem sx={{ justifyContent: 'center' }}>
          <Typography
            variant="p"
            sx={{
              fontFamily: 'monospace',
              fontWeight: 400,
              color: 'inherit',
              textDecoration: 'none',
              mr: 1,
            }}
          >
            Sending "keypoints3D" via MQTT:
          </Typography>
          <Chip label={sendingKeypointsInfo} style={{ fontWeight: '500px' }} />
          {MsgCount > 0 && MsgCount}
        </ListItem>
      </List>
    </>
  )
}

export default KeypointsSender
