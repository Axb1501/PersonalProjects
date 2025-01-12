import pandas as pd
import requests
import time
import subprocess
import os
import re


def parse_slither_output(output):
    # Split the text into lines
    lines = output.split('\n')

    # Initialize an empty list to hold the lines that start with '-' but do not contain 'From'
    filtered_lines = []

    # Iterate over each line in the input text
    for line in lines:
        # Strip leading and trailing whitespace
        stripped_line = line.strip()
    
        # Check if the line starts with '-' and does not contain 'From'
        if stripped_line.startswith('-') and 'From' not in stripped_line:
            # Add the line to the filtered list
            filtered_lines.append(stripped_line)

    # Format the output with each function on a new line, preceded and followed by "# function #"
    if filtered_lines:
        output_text = "\n#\n\n" + "\n\n#\n\n".join(filtered_lines)
    else:
        output_text = ""

    return output_text


def get_functions():

    # Start the Slither process with subprocess.Popen
    process = subprocess.Popen(
        ["slither", "split.sol", "--print", "contract-summary"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for the process to complete and capture the stdout and stderr
    stdout, stderr = process.communicate()

    # Check if Slither command was successful
    if process.returncode == 0:
        # Extract relevant information from the Slither stdout
        extracted_info = parse_slither_output(stderr)
    
        # Append the extracted information to the file instead of printing
        with open('C:/Users/aaron/myproject/myproject/smartcontract.txt', 'a') as file:
            file.write(extracted_info + "\n")
    
       
  

def extract_constructors():
    # Split the output into lines for easier processing
     # Start the Slither process with subprocess.Popen
    process = subprocess.Popen(
        ["slither", "split.sol", "--print", "constructor-calls"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for the process to complete and capture the stdout and stderr
    stdout, stderr = process.communicate()
    
    lines = stderr.split('\n')

    # List to store constructors and their codes
    constructors = []

    # Temporary storage for constructor code
    constructor_code = []
    capture = False

    # Iterate over each line to find and extract constructor definitions
    for line in lines:
        # Check if the line contains the constructor keyword
        if 'constructor()' in line:
            capture = True
            constructor_code.append(line)
        elif capture:
            constructor_code.append(line)
            # Check if the line is the closing of a constructor
            if line.strip() == '}':
                capture = False
                constructors.append("\n".join(constructor_code))
                constructor_code = []

    # Append the constructors and their codes to the file
    with open('C:/Users/aaron/myproject/myproject/smartcontract.txt', 'a') as file:
        for constructor in constructors:
            file.write("#\n\n" + constructor + "\n#\n")


# Etherscan API key
api_key = 'P8D14TDX41G1GVIG43SE13MUKMQRS3TH8V'

# Load CSV file into a pandas DataFrame
csv_path = "C:/Users/aaron/myproject/myproject/trainingdata.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_path)

# Define the output file path
output_file = "split.sol"

# Limit the loop to the first 100 rows
for index, row in df.head(2).iterrows():
    contract_address = row['ContractAddress']
    tx_hash = row['Txhash']

    api_url = 'https://api.etherscan.io/api'
    params = {
        'module': 'contract',
        'action': 'getsourcecode',
        'address': contract_address,
        'apikey': api_key,
    }

    # Send request to Etherscan API
    response = requests.get(api_url, params=params)
    data = response.json()

    # Check if response is successful
    if data['status'] == '1':
        # Extract source code
        source_code = data['result'][0]['SourceCode']
        
        # Write source code to file
        with open(output_file, 'w') as f:
            f.write(source_code)
        
        print(f"Source code for contract {contract_address} written to {output_file}")
        # Call the function extract_constructors
        extract_constructors()
        
        # Call the function get_functions
        get_functions()
    else:
        print(f"Failed to fetch source code for contract {contract_address}")
    
    # Add some delay to avoid hitting API rate limits
    time.sleep(1)  # Sleep for 1 second





# Terminal output as a multi-line string
terminal_output2 = """
'solc --version' running
'solc split.sol --combined-json abi,ast,bin,bin-runtime,srcmap,srcmap-runtime,userdoc,devdoc,hashes --allow-paths .,C:/Users/aaron/myproject/myproject' running
INFO:Printers:
Contract SafeMath
+----------+-----------+
| Function | Modifiers |
+----------+-----------+
| add      |        [] |
| sub      |        [] |
| sub      |        [] |
| mul      |        [] |
| div      |        [] |
| div      |        [] |
+----------+-----------+
INFO:Printers:
Contract IUniswapV2Factory
+------------+-----------+
| Function   | Modifiers |
+------------+-----------+
| createPair |        [] |
+------------+-----------+
INFO:Printers:
Contract IUniswapV2Router02
+----------------------------------------------------+-----------+
| Function                                           | Modifiers |
+----------------------------------------------------+-----------+
| swapExactTokensForETHSupportingFeeOnTransferTokens |        [] |
| factory                                            |        [] |
| WETH                                               |        [] |
| addLiquidityETH                                    |        [] |
+----------------------------------------------------+-----------+
INFO:Printers:
Contract AI71
+-------------------------------------+-----------------+
| Function                            |       Modifiers |
+-------------------------------------+-----------------+
| constructor                         |              [] |
| owner                               |              [] |
| renounceOwnership                   |   ['onlyOwner'] |
| transferOwnership                   |   ['onlyOwner'] |
| _msgSender                          |              [] |
| totalSupply                         |              [] |
| balanceOf                           |              [] |
| transfer                            |              [] |
| allowance                           |              [] |
| approve                             |              [] |
| transferFrom                        |              [] |
| constructor                         |              [] |
| name                                |              [] |
| symbol                              |              [] |
| decimals                            |              [] |
| totalSupply                         |              [] |
| balanceOf                           |              [] |
| transfer                            | ['lockTheSwap'] |
| allowance                           |              [] |
| approve                             |              [] |
| transferFrom                        | ['lockTheSwap'] |
| tokenFromReflection                 |              [] |
| removeAllFee                        |              [] |
| restoreAllFee                       |              [] |
| _approve                            |              [] |
| _transfer                           | ['lockTheSwap'] |
| swapTokensForEth                    | ['lockTheSwap'] |
| sendETHToFee                        |              [] |
| setTrading                          |   ['onlyOwner'] |
| manualswap                          | ['lockTheSwap'] |
| manualsend                          |              [] |
| blockBots                           |   ['onlyOwner'] |
| unblockBot                          |   ['onlyOwner'] |
| _tokenTransfer                      |              [] |
| _transferStandard                   |              [] |
| _takeTeam                           |              [] |
| _reflectFee                         |              [] |
| receive                             |              [] |
| _getValues                          |              [] |
| _getTValues                         |              [] |
| _getRValues                         |              [] |
| _getRate                            |              [] |
| _getCurrentSupply                   |              [] |
| setFee                              |   ['onlyOwner'] |
| setMinSwapTokensThreshold           |   ['onlyOwner'] |
| toggleSwap                          |   ['onlyOwner'] |
| setMaxTxnAmount                     |   ['onlyOwner'] |
| setMaxWalletSize                    |   ['onlyOwner'] |
| excludeMultipleAccountsFromFees     |   ['onlyOwner'] |
| slitherConstructorVariables         |              [] |
| slitherConstructorConstantVariables |              [] |
+-------------------------------------+-----------------+
INFO:Slither:split.sol analyzed (7 contracts)
"""

def get_modifiers(terminal_output2):
    # Regular expression to match lines with function names and their non-empty modifiers
    # It captures the function name and the modifiers, ignoring lines with empty brackets []#
    pattern = re.compile(r'\|\s*([^|]+?)\s*\|\s*\[([^\]]+?)\]\s*\|')
    matches = pattern.findall(terminal_output2)

    # Print each function with its modifiers, excluding those with empty modifiers []
    for match in matches:
        print(f"Function: {match[0].strip()}, Modifiers: [{match[1]}]")



def deduplicate_smart_contract_file(filename):
    try:
        # Read the content from the file
        with open(filename, 'r') as file:
            content = file.read()
        
        # Split the content by '# function #' to extract the relevant sections
        parts = content.split("#")
        unique_parts = set()  # Use a set to avoid duplicates

        # Process each part and add it to the set
        for part in parts:
            stripped_part = part.strip()
            if stripped_part:
                unique_parts.add(stripped_part)

        # Write the unique parts back to the file, with formatting
        with open(filename, 'w') as file:
            for part in unique_parts:
                file.write("#\n\n" + part + "\n\n")
        
        print(f"Updated {filename} with unique entries.")
    
    except IOError as e:
        print(f"An error occurred while accessing {filename}: {e}")

# Example usage
extract_constructors()

# Example usage:
deduplicate_smart_contract_file('C:/Users/aaron/myproject/myproject/smartcontract.txt')
