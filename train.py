from datasets import load_dataset
import torch 
from transformers import AutoTokenizer
from model import TranslationDataset, TranslationModel
from torch.utils.data import DataLoader
# Load English-French translation split from IWSLT2017

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = load_dataset("iwslt2017", "iwslt2017-fr-en", split="train")

dataset = dataset.filter(lambda x: len(x['translation']['fr'].split()) < 30 and len(x['translation']['en'].split()) < 30)

pairs = [(x["translation"]["en"], x["translation"]["fr"]) for x in dataset]

tokenizer = AutoTokenizer.from_pretrained("t5-small")

tokens_size = tokenizer.vocab_size

dataset = TranslationDataset(pairs[:50], tokenizer)


# params
T1 = 32
T2 = 32
heads = 256
batch_size = 16
no_of_blocks = 6



dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

translationModel = TranslationModel(heads,tokens_size,T1,T2,no_of_blocks).to(device)

optimizer = torch.optim.AdamW(translationModel.parameters(),lr=1e-3)

for i in range(100000):
  batch = next(iter(dataloader))
  context = batch["context"].to(device)
  decoder_input = batch["decoder_input"].to(device)
  target = batch["target"].to(device)
  logits, loss  = translationModel(context,decoder_input,target)
  if i%10==0:
      print(f"At step {i} loss is: {loss.item()}")
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  # xm.optimizer_step(optimizer)
  optimizer.step()
  if i % 1000 == 0 and i > 0:
        torch.save(translationModel.state_dict(), f"model_step_{i}.pt")
torch.save(translationModel.state_dict(), "final_model.pt")