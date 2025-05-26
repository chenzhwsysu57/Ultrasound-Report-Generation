debug info

memory.shape
shape = torch.Size([16, 98, 512]) [batch_size, 49*2, hidden_size]

mask
shape = torch.Size([16, 1, 98])

for l, x in zip(self.linears, (query, key, value)):
    print(l, x.shape)