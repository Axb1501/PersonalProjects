import React, { useState } from 'react';
import axios from 'axios';
import Header from './constants/Header';

const Page2 = () => {
  const [inputText, setInputText] = useState('');
  const [textWithIsCool, setTextWithIsCool] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleAddIsCool = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/add_is_cool/', { text: inputText });
      setTextWithIsCool(response.data.text_with_is_cool);
    } catch (error) {
      console.error('Error adding "is cool":', error);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setInputText(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  const triggerFileInput = () => {
    document.getElementById('fileInput').click();
  };

  return (
    <div>
      <Header />
      <div style={{ display: 'flex', justifyContent: 'space-around', margin: '20px' }}>
        <div style={{ width: '40%' }}>
          <div style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between' }}>
            
            <input
              type="file"
              id="fileInput"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>
          <textarea 
            style={{
              width: '100%', 
              height: '150px', 
              marginBottom: '10px'
            }} 
            value={inputText} 
            onChange={handleInputChange}
          />
        </div>
        <div style={{ width: '50%', border: '1px solid black', padding: '10px', minHeight: '200px' }}>
          <textarea 
            style={{ 
              width: '100%', 
              height: '200px', 
              border: 'none', 
              resize: 'none'
            }} 
            value={textWithIsCool} 
            readOnly 
          />
        </div>
      </div>
    </div>
  );
};

export default Page2;


