import CssBaseline from '@mui/material/CssBaseline'
import Container from '@mui/material/Container'

import CanvasWrapper from './components/CanvasWrapper'
import NavBar from './components/NavBar'

// import logo from './logo.svg'
import './App.css'

function App() {
  return (
    <div className="App">
      <CssBaseline />
      <NavBar />
      <Container maxWidth="100%">
        <CanvasWrapper />
      </Container>
    </div>
  )
}

export default App
