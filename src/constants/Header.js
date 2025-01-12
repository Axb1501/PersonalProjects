import React from 'react';
import { Link } from 'react-router-dom';

function Header() {
  return (
    <header style={headerStyle}>
      <nav>
        <ul style={navStyle}>         
          <li style={{ ...listItemStyle, marginLeft: 'auto' }}><Link to="/" style={linkStyle}>Code</Link></li>
          <li style={listItemStyle}><Link to="/page2" style={linkStyle}></Link></li>
        </ul>
      </nav>
    </header>
  );
}

export default Header;

// Styles
const headerStyle = {
  backgroundColor: 'black', // Change the background color to black
  fontSize: '24px', // Increase the font size of the header
  textAlign: 'center', // Center the text horizontally
  margin: '0', // Reset margin
  padding: '0', // Reset padding
};

const navStyle = {
  display: 'flex',
  justifyContent: 'flex-end', // Align items to the right
  padding: '10px',
};

const listItemStyle = {
  listStyleType: 'none',
  marginRight: '100px', // Add margin to separate the links
};

const linkStyle = {
  color: 'white', // Change the color of the links to white
  textDecoration: 'none', // Remove the underline from the links
};
