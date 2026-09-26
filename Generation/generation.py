import torch
from torch import nn
from model import CharacterTransformer
from tokenizer import Tokenizer




model = CharacterTransformer()
tokenizer = Tokenizer()



max_tokens = 115
user_input = str(input("You: "))
text = user_input.lower()

with torch.no_grad():
    for _ in range(max_tokens):
        
        context = text[-32:]
        #This takes the last 32 letters of the text (I know sad right) #But we need it to maximize preformance and make it not take a gillion years
        #As well as following the rule of the self.positonal_emeddings() ~~ 100 characters
        
        
        tokenizer_text = torch.tensor([tokenizer.tokenize(context)])

        model_output = model(tokenizer_text)

   
        next_token_id = model_output[0, -1, :].argmax(dim=-1).item()
        next_char = tokenizer.decode([next_token_id])

       
        text += next_char

print(text)
