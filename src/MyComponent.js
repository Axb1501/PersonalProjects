import React, { useState } from 'react';

function MyComponent() {
  const [sequence, setSequence] = useState('');

  const handleChange = (event) => {
    setSequence(event.target.value);
  };

  const handleReverseSequence = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/reverse-sequence', {
        method: 'POST', // Changed to POST
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence }),
      });

      if (response.ok) {
        const reversedSequence = await response.json();
        setSequence(reversedSequence);
      } else {
        // Handle errors appropriately, e.g., display an error message to the user
        console.error('Error fetching reversed sequence:', response.statusText);
      }
    } catch (error) {
      console.error('Error fetching reversed sequence:', error);
    }
  };

  return (
    <div>
      <input type="text" value={sequence} onChange={handleChange} />
      <button onClick={() => handleReverseSequence()}>Reverse Sequence</button>
    </div>
  );
}

export default MyComponent;
