import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Page1 from './Page1';
import Page2 from './Page2';
import Register from './Register'; // Import your Register component
import Login from './Login'; // Import your Login component
import './App.css'

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Page1 />} />
          <Route path="/page2" element={<Page2 />} />
          <Route path="/register" element={<Register />} /> {/* Route for Register component */}
          <Route path="/login" element={<Login />} /> {/* Route for Login component */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;

