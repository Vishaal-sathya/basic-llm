
from training.dataset import create_data_loader
from llm.llm_config import LLM_CONFIG


def create_loaders(file_path, train_ratio, batch_size, max_length, stride, drop_last=True, shuffle=True, num_workers=0):
    
    with open(file_path,'r', encoding='utf-8') as file:
        text_data = file.read()

    
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]



    train_loader = create_data_loader(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_workers
    )

    val_loader = create_data_loader(
        val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return train_loader, val_loader