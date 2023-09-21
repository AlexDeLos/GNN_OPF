def train(data, model, optimizer, criterion, device='cpu'):
    model.train()
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def evaluate(data, model, criterion, device='cpu'):
    model.eval()
    model.to(device)
    out = model(data)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss

def test(data, model, device='cpu'):
    model.eval()
    model.to(device)
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return acc