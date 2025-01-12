import React, { useState } from 'react';
import axios from 'axios';
import Header from './constants/Header';

const Page1 = () => {
    const [inputText, setInputText] = useState('');
    const [secondInputText, setSecondInputText] = useState('');
    const [isSecondInputEditable, setIsSecondInputEditable] = useState(true);
    const [outputText, setOutputText] = useState('');

    const handleApiCall = async (url, payload) => {
        try {
            const response = await axios.post(url, payload);
            return response.data;
        } catch (error) {
            console.error('API error:', error);
            return '';
        }
    };

    const handleFirstInputChange = (e) => {
        setInputText(e.target.value);
    };

    const handleSecondInputChange = (e) => {
        setSecondInputText(e.target.value);
    };

    const handleFirstRunButtonClick = async () => {
        const data = await handleApiCall('http://127.0.0.1:8000/reverse_text/', { text: inputText });
        setSecondInputText(data.reversed_text);
        setIsSecondInputEditable(true);
    };

    const handleSecondRunButtonClick = async () => {
        const data = await handleApiCall('http://127.0.0.1:8000/test/', { text: secondInputText });
        setOutputText(data.combined_text);
    };

    const handleDownloadButtonClick = () => {
        const element = document.createElement("a");
        const file = new Blob([outputText], { type: 'text/plain' });
        element.href = URL.createObjectURL(file);
        element.download = "output.sol";
        document.body.appendChild(element); // Required for this to work in FireFox
        element.click();
        document.body.removeChild(element);
    };

    return (
        <div>
            <Header />
            <div style={{ padding: '20px' }}>
                <div style={{ marginBottom: '10px' }}>
                    <input
                        type="text"
                        placeholder="Please enter Smart contract functionality"
                        value={inputText}
                        onChange={handleFirstInputChange}
                        style={{ width: '40%', marginRight: '10px' }}
                    />
                    <button onClick={handleFirstRunButtonClick} style={{ width: '10%', marginRight: '10px' }}>Run</button>
                    <button onClick={handleDownloadButtonClick} style={{ width: '10%', marginRight: '10px' }}>Download</button>
                    <button style={{ width: '10%' }}>Security</button>
                </div>
                <div style={{ display: 'flex' }}>
                    <div
                        style={{
                            flex: 1,
                            marginRight: '10px',
                            padding: '10px',
                            border: '1px solid black',
                            minHeight: '300px',
                        }}
                    >
                        Editor <button onClick={handleSecondRunButtonClick} style={{ marginLeft: '10px' }}>Run</button>
                        <textarea
                            placeholder="Editor"
                            value={secondInputText}
                            onChange={handleSecondInputChange}
                            style={{ width: '100%', height: '250px' }}
                            disabled={!isSecondInputEditable}
                        />
                    </div>
                    <div
                        style={{
                            flex: 1,
                            padding: '10px',
                            border: '1px solid black',
                            minHeight: '300px',
                            overflowY: 'auto'
                        }}
                    >
                        <pre>{outputText}</pre> {/* This box will display the combined output */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Page1;
