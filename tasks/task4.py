import torch
import torch.nn as nn
from sentence_transformer import MTLTransformer, SentencesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Model initialization
model = MTLTransformer()

# Loss functions for training
classification_loss_func = nn.CrossEntropyLoss()
sentiment_loss_func = nn.CrossEntropyLoss()

# Optimizer initialization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  

# Number of epochs 
num_epochs = 100

# Dataloader
train_dataset = SentencesDataset("sentences.csv")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)


# Training loop
for epoch in tqdm(range(num_epochs)):
    loss_per_epoch = 0 
    for inputs, class_labels, sentiment_labels in train_loader:

        # Zero out optimizer before every pass        
        optimizer.zero_grad()

        # Forward pass
        output_class, output_sentiment = model(inputs)

        # Calculate loss based on both tasks
        sentiment_loss = sentiment_loss_func(output_sentiment, torch.Tensor(sentiment_labels)[..., None])
        class_loss = classification_loss_func(output_class, torch.LongTensor(class_labels))
        total_loss = sentiment_loss + class_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        loss_per_epoch += total_loss.item()