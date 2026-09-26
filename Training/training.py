import torch
from torch import nn
from model import CharacterTransformer
from tokenizer import Tokenizer




tokenizer = Tokenizer()


model = CharacterTransformer()

try:
  model.load_state_dict(torch.load("character_transformer.pt", map_location=torch.device('cpu')))

  print("Weights loaded successfully!")
except Exception as e:
  print(f"ERROR AS {e}")
  print("character_transformer.pt not found, switching to pre-existing weights")

#Setting Up Training

#Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

#Optimizer function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


input_text = "abcdefg"
target_text = "bcdefgh"

epoches = 100000


that = "To be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles shall i compare thee to a summers day"


try:
  with open("corpus_clean.txt", "r") as f:
      text_data = f.read().lower().replace("\n"," ")
except Exception as e:
  print("corpus.txt not found, switching to small alternitive example")

  text_data = that.lower().replace("\n"," ").replace("'"," ")



start_time = time.time()

batch_size = 32



for epoch in range(epoches):
  str_thing = []
  str_out = []  

  optimizer.zero_grad()
  block_size = 32

  max_start = len(text_data) - block_size - 1


  for _ in range(batch_size):

    start_idx = random.randint(0, max_start)


    in_the_model = text_data[start_idx:start_idx + block_size]
    should_be_out_the_model = text_data[start_idx + 1:start_idx + block_size + 1]

    str_thing.append(in_the_model)
    str_out.append(should_be_out_the_model)
  
  
  tokenized_inputs = [tokenizer.tokenize(s) for s in str_thing]
  tokenized_targets = [tokenizer.tokenize(s) for s in str_out]

  thatrandomresult = model.forward(torch.tensor(tokenized_inputs))
  theactualresult = torch.tensor(tokenized_targets) 




  loss = loss_fn(thatrandomresult.transpose(1,2),theactualresult)

  loss.backward()
  if epoch > epoches - 5:
     
        end_time = time.time()
        print(start_time - end_time)

  optimizer.step()


  if epoch % 5000 == 0:
    print(f"LOSS:{loss} EPOCH:{epoch}")
    pred_ids = thatrandomresult.argmax(dim=-1)[0].tolist()
    print(f"TEXT:{tokenizer.decode(pred_ids)}")




#So we need to first pritn the outputs: Aka the number's not the words,

#And then we need to print out the targer (basically just a tokenized version of the item.)
torch.save(model.state_dict(), "character_transformer.pt")
print("Model weights successfully saved to character_transformer.pt!")
