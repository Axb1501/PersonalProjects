import os
from openai import OpenAI
import re
from .tt import eng_to_solidity
from .code_test import code_test

client = OpenAI(
    # This is the default and can be omitted
    api_key = "#key goes here#"
)


def Generate(user_input):
# Define the prompt related to a smart contract
    prompt = f"Describe the smart contract tasks you'd like to implement.\n\nChatGPT: Provide a brief description of the smart contract functionality or tasks you'd like to implement, including any interfaces, state variables, constructors, functions, and modifiers.\n\nUser: {user_input}\n"

    # Create the OpenAI API client
    chat_completion = client.chat.completions.create(
     model="gpt-3.5-turbo",
     messages=[
         {
                "role": "system",
                "content": "Bullet point a description of the functions, events and constructors that the smart contract from user input needs make sure include every possible function, event or constructor needed, do not include error handling or state variables. If the prompt is invalid return 'Error invalid input'"
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )



#    Extract text from the completion
    chat_text = chat_completion.choices[0].message.content

    return chat_text


def test_and_combine(chat):
    
    error = "Error invalid input"
    # Extract lines starting with '-'
    
    if chat == "Error invalid input":
        return(error)
    
    lines = re.findall(r'^\s*-\s.*?(?=\n|$)', chat, re.MULTILINE)

    # Convert each point in the lines array using eng_to_solidity function 
    converted_lines = []
    for line in lines:
        # Add comment above the converted line
        comment = "// English: " + line.strip()[1:]
        solidity_code = eng_to_solidity(line.strip()[1:])
        if solidity_code is not None:
            converted_lines.append(code_test(comment + solidity_code))
    
    
    # Combine the outputs into a single string
    combined_output = '\n'.join(converted_lines)

    # Print the combined output
    combined_output = '''// SPDX-License-Identifier: UNLICENSED
    pragma solidity ^0.8.20;
    contract MyContract {
        ''' + combined_output + '''
    }'''


    return combined_output

