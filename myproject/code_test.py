from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key = "#key goes here#"
)

def code_test(prompt):
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Determine if the solidity code satisfies the criteria marked by //. If it doesnt output a corrected version that does with the //comment, otherwise if it does output the //comment and the code unchanged"
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return chat_completion.choices[0].message.content

print(code_test("// English:  Function: Get the balance of an address function checkBalance (address account )public view returns (uint256 ){return balances [msg .sender ];}"
))