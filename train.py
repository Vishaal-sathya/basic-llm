import torch
import tiktoken
from llm.llm_config import LLM_CONFIG
from llm.llm_architecture import LLMModel
from training.prepare_data import create_loaders
from training.train_model import train_model


file_path = "the-verdict.txt"
train_ratio = 0.9
train_loader, val_loader = create_loaders(file_path, train_ratio, 
                                          batch_size=2, 
                                          max_length=256, 
                                          stride=256)

torch.manual_seed(123)
model = LLMModel(LLM_CONFIG)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
tokenizer = tiktoken.get_encoding('gpt2')


train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5,eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)