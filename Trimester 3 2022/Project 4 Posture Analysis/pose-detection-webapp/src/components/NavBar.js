import * as React from 'react'
import AppBar from '@mui/material/AppBar'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import Container from '@mui/material/Container'

import logo from '../redback-logo.png'

function NavBar() {
  return (
    <AppBar position="static">
      <Container
        maxWidth="none"
        sx={{
          m: 0,
          width: '100%',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <Toolbar disableGutters>
          <img src={logo} alt="logo" style={{ width: '80px', marginRight: '10px' }} />
          <Typography
            variant="h6"
            noWrap
            sx={{
              display: { xs: 'none', md: 'flex' },
              fontFamily: 'monospace',
              fontWeight: 500,
              //   letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            Redback Operations / Data Analytics
          </Typography>
        </Toolbar>
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            noWrap
            sx={{
              display: { md: 'flex', justifyContent: 'flex-end' },
              fontFamily: 'monospace',
              fontWeight: 600,
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            Pose Detection & Estimation
          </Typography>
        </Toolbar>
      </Container>
    </AppBar>
  )
}
export default NavBar
