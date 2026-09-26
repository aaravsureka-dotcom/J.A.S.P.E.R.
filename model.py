import re
import torch
import torch.nn as nn
import math
import random




device = "CPU"
other_device = "GPU"

class CharacterTransformer(nn.Module):
  def __init__(self,vocab_size=41,emedding_dimensions=32):
    super().__init__()
    self.embedding_layer = nn.Embedding(vocab_size, emedding_dimensions)


    self.position_embedding_layer = nn.Embedding(100, 32)

    self.layer_norm1 = nn.LayerNorm(emedding_dimensions)
    self.layer_norm2 = nn.LayerNorm(emedding_dimensions)
    #ok so layer norm turns large numbers into numbers between a range of -1 to 1, it is often used in tranformers and nerual networks to scale down the large numbers into managable ones
    #Normilization usually happens before attention and before the thinkking (feed forward network) this happens so the AI can train much faster and never really
    #have to worry about it's numbers blowing up in size
    #We usually have 2 normilization layers beacuse one is used before attention and the other is used before the thinking, so both of them have diffrent
    #jobs, and they are tuning diffrent types of imformation, because when layer1 normilizes the values before, it also learns about the weights and bias
    #For that specific before attention, but after attention the numbers are commpletly diffrent so layer 1 cannot do it again. In simple terms
    #2 layers are needed because, Layer 1 learn how to prepare numbers for Attetnion (like preparing a dog for a bath) and layer 2 prepares the number's
    #for the ffn Like preparing a baby for a bath, both of them are essentially doign the same job, but for diffrent types of things


    self.query_layer = nn.Linear(32,32,bias=False)
    self.key_layer = nn.Linear(32,32,bias=False)
    self.value_layer = nn.Linear(32,32,bias=False)
    #NOW here comes the attetnion mechanisim, so the attention's mechanisim has 1 main job, and that job is to figure out how a character is related to
    #another character so that the AI can understand what is actually happenign, see letters can change based on what is happening aroud it such as
    #The robbers robbed a bank, the fisherman sat on the river bank, see without attention AI would look at bank and say HEY it's bank, okay lets save
    #that as that bank, with attention it looks, HEY there is a f i s h e r m a n that means that this bank is about the river (looking at the river)
    #See to do this there are 3 main steps that happens: Query, Key and Value
    #Query: This is kind of the searcher, this guy says HEY I AM SERACHING FOR X
    #Key: He goes like HEY BROTHER I am a (noun)/(verb)/action like the actual question, like this is who I am and what I have to offer
    #Basically meaning that Hey I am [character] and I have [x] to offer
    #SO using this we can get the matchmaking, using this Q and K we can find the probabillities of thing using matrix multiplicaton
    #SO basically what hpapens is that query goes like I am the letter A and key goes like I am the letter N and they multiply and say
    # HEY thast PERFECT we are correlated!
    #Then with those values we multiply it by the last amount which is the KEY
    #The key has a very importatn  job
    self.num_heads = 4

    q_heads = emedding_dimensions // self.num_heads
    #See many people find this part very confusing, this is 3 linear layers (AKA a nerual network layer, the first one takes 32 inputs (numbers ) then spirts out 32, same thing 3 times )
    #For more reasoning and the AI is smarter, its spitting 32 numbers in the end because to find which letter is it saying, we need 32 NUMBERS which will propose to the newest
    #nukber
    self.return_layer = nn.Linear(32,41,bias=False)


    self.ff_layer = nn.Sequential(
        nn.Linear(32,4 * emedding_dimensions),
        nn.GELU(),
        nn.Linear(4 * emedding_dimensions ,emedding_dimensions)
    )

    self.out_projection = nn.Linear(32,32)


  def forward(self,inx):
    actualitems = self.embedding_layer(inx)

    seq_len = inx.shape[1]

    pos_indices = torch.arange(seq_len, device=inx.device)

    thatstuff = self.position_embedding_layer(pos_indices)

    x = actualitems + thatstuff

    clean_x = self.layer_norm1(x)

    qlayer = self.query_layer(clean_x)
    klayer = self.key_layer(clean_x)
    vlayer = self.value_layer(clean_x)

    b,t,c = qlayer.shape

    no_heads = self.num_heads
    head_size = c // no_heads

    qlayer = qlayer.view(b,t,no_heads,head_size)
    klayer = klayer.view(b,t,no_heads,head_size)
    vlayer = vlayer.view(b,t,no_heads,head_size)

    qlayer = qlayer.transpose(1,2)
    klayer = klayer.transpose(1,2)
    vlayer = vlayer.transpose(1,2)
    #Here they now have gone through 3 layers and now it's time to calculate the score (percentages to see how much a letter realyl matters)

    score = (qlayer @ klayer.transpose(-2, -1)) / math.sqrt(head_size)

    causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=inx.device), diagonal=1)

    score = score.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))



    result = torch.softmax(score, dim=-1)

    blendtime = result @ vlayer

    blendtime = blendtime.transpose(1, 2).contiguous().view(b, t, c)

    out = self.out_projection(blendtime)

    x = x + out

    clean_x = self.layer_norm2(x)

    out_put_final = self.ff_layer(clean_x)

    x = x + out_put_final




    finalthing = self.return_layer(x)



    return finalthing
#lets say we have a letter such as ABCDEFG  right, right now the tokenizer will return all the index value's of thata item such as [0,1,2,3,4,5,6], but now that we know the index value, we need the
#actual value of the item, the 32 numbers on the thing so that the transformer can reason, but we cannot just give it the 32 numebrs, we have to give the actual positon as well, because ABCDEF
#will also look ike BDACEF as well, so to give it an index, the forward method first counts how long the thing is, by doing inx.shape[1] BECAUSE it usually will return in 2d tensors so [0] would
#Just say how many words there are, sso then we need to arrange the index sequence from first letter [0] to last letter [-1] using the torch.arrange method, because our position emmedding layer
#is the same as our emedding dimensions, each number is added up 32 times just like that, so when we add it we change it's values to mkake

vocab_size = 41

emedding_dimensions = 32
