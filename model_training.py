#Training Step

def make_train_step(model, loss_fn, optimizer):

  def perform_train_step(X, y):

    #Set model train
    model.train()

    yhat = model(X)
    
    loss = loss_fn(yhat, y)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


  return perform_train_step


make_train_step(model, loss_fn, optimizer)


n_epochs = 1000

losses = []

for epochs in range(n_epochs):
  ##inner loop
  mini_batches_losses = []
  for x_batch, y_batch in train_loader:
    #move datasets to device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)


    #perform one train step and return the loss
    mini_batch_loss = train_step(x_batch, y_batch)
    mini_batches_losses.append(mini_batch_loss)


  # Compute the average loss
  loss = np.mean(mini_batches_losses)

  losses.append(loss)

  
  
  
  
  
  
  def mini_batch(device, data_loader, step):
  mini_batches_losses = []
  for x_batch, y_batch in data_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    mini_batch_loss = step(x_batch, y_batch)
    mini_batches_losses.append(mini_batch_loss)

  loss = np.mean(mini_batches_losses)
  return loss
