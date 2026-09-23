  import re
  import torch
  import torch.nn as nn
  import random


  class CharacterTransformer(nn.Module):
    def __init__(self,vocab_size=41,emedding_dimensions=32):
      super().__init__()
      self.embedding_layer = nn.Embedding(vocab_size, emedding_dimensions)

      self.position_embedding_layer = nn.Embedding(100, 32)


      self.query_layer = nn.Linear(32,32,bias=False)
      self.key_layer = nn.Linear(32,32,bias=False)
      self.value_layer = nn.Linear(32,32,bias=False)
      #See many people find this part very confusing, this is 3 linear layers (AKA a nerual network layer, the first one takes 32 inputs (numbers ) then spirts out 32, same thing 3 times )
      #For more reasoning and the AI is smarter, its spitting 32 numbers in the end because to find which letter is it saying, we need 32 NUMBERS which will propose to the newest
      #nukber
      self.return_layer = nn.Linear(32,41,bias=False)

    def forward(self,inx):
     

      actualitems = self.embedding_layer(inx)

      seq_len = inx.shape[1]

      mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

      pos_indices = torch.arange(seq_len)

      thatstuff = self.position_embedding_layer(pos_indices)

      x = actualitems + thatstuff


      qlayer = self.query_layer(x)
      klayer = self.key_layer(x)
      vlayer = self.value_layer(x)
      

      #Here they now have gone through 3 layers and now it's time to calculate the score (percentages to see how much a letter realyl matters)

      score = qlayer @ klayer.transpose(-2,-1)
      score = score + mask
      result = torch.softmax(score,dim=-1)

      blendtime = result @ vlayer

      finalthing = self.return_layer(blendtime)

      

      return finalthing
  #lets say we have a letter such as ABCDEFG  right, right now the tokenizer will return all the index value's of thata item such as [0,1,2,3,4,5,6], but now that we know the index value, we need the
  #actual value of the item, the 32 numbers on the thing so that the transformer can reason, but we cannot just give it the 32 numebrs, we have to give the actual positon as well, because ABCDEF
  #will also look ike BDACEF as well, so to give it an index, the forward method first counts how long the thing is, by doing inx.shape[1] BECAUSE it usually will return in 2d tensors so [0] would
  #Just say how many words there are, sso then we need to arrange the index sequence from first letter [0] to last letter [-1] using the torch.arrange method, because our position emmedding layer
  #is the same as our emedding dimensions, each number is added up 32 times just like that, so when we add it we change it's values to mkake

  vocab_size = 41

  emedding_dimensions = 32

  class Tokenizer:
    def __init__(self):
      self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")


      self.charnum = {x:char for char,x in enumerate(self.chars)}

      self.numchar = {char:x for char,x in enumerate(self.chars) }
    def tokenize(self,string):
      return [self.charnum[char] for char in string.lower()]
    def decode(self, token_ids):
        return "".join([self.numchar[item] for item in token_ids])

  tokenizer = Tokenizer()


  model = CharacterTransformer()
 


 


  #Setting Up Training 


  #Loss Function
  loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

  #Optimizer function
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


  input_text = "abcdefg"
  target_text = "bcdefgh"

  epoches = 15000


  with open("corpus.txt", "r") as f:
      text_data = f.read().lower().replace("\n"," ")



  for epoch in range(epoches):
    optimizer.zero_grad()
    block_size = 32
    
    max_start = len(text_data) - block_size - 1
    start_idx = random.randint(0, max_start)

    in_the_model = text_data[start_idx:start_idx + block_size]
    should_be_out_the_model = text_data[start_idx + 1:start_idx + block_size + 1]

    thatrandomresult = model.forward(torch.tensor([tokenizer.tokenize(in_the_model)]))
    theactualresult = (torch.tensor([tokenizer.tokenize(should_be_out_the_model)]))
    

   

    loss = loss_fn(thatrandomresult.transpose(1,2),theactualresult)

    loss.backward()

    optimizer.step()
    if epoch % 200 == 0:
      print(f"LOSS:{loss} EPOCH:{epoch}")
      pred_ids = thatrandomresult.argmax(dim=-1)[0].tolist()
      print(f"TEXT:{tokenizer.decode(pred_ids)}")
    


  #So we need to first pritn the outputs: Aka the number's not the words, 

  #And then we need to print out the targer (basically just a tokenized version of the item.)




