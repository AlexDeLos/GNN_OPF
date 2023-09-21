# Single graph training functions
def train_single_graph(data, model, optimizer, criterion, device='cpu'):
    model.train()
    model.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def evaluate_single_graph(data, model, criterion, device='cpu'):
    model.eval()
    model.to(device)
    out = model(data)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss

def test_single_graph(data, model, device='cpu'):
    model.eval()
    model.to(device)
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return acc

#############################################################################################################
# Multi graph training functions

def train_epoch(model, loader, optimizer, criterion, device='cpu'):
  model.train()
  model.to(device)
  optimizer.zero_grad()

  total_loss = 0.0

  for batch in loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, batch.y.long())
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss / len(loader)


def evaluate_epoch(model, loader, criterion, device='cpu'):
    model.eval()
    model.to(device)

    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = criterion(output, batch.y.long())
        total_loss += loss.item()

    return total_loss / len(loader)


def test_epoch(model, loader, device='cpu'):
    model.eval()
    model.to(device)

    correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        pred = output.argmax(dim=1)
        correct += int((pred == batch.y).sum())
        total_samples += batch.num_graphs * batch.num_nodes

    accuracy = correct / total_samples

    return accuracy