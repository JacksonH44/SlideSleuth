from vae import VAE
from vae import load_csv_files
import matplotlib.pyplot as plt

LATENT_SPACE_DIM = 32
TRAIN_DIR = '../outputs/HNE_features/train'
TEST_DIR = '../outputs/HNE_features/test'

BATCH_SIZE = 32
NUM_EPOCHS = 100

def plot_loss(history, metric, save_file):
  """A function that plots the loss metric over time for a vae training session

  Args:
      history (History): The history of a fitted model
  """
  plt.plot(
    history.epoch, 
    history.history[metric], 
    color='deepskyblue', 
    label = f'Train {metric}'
  )
  plt.plot(
    history.epoch,
    history.history[f'val_{metric}'],
    color='deepskyblue',
    label=f'Valid {metric}',
    linestyle='--'
  )
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(f'{metric}')
  plt.savefig(save_file)
  plt.legend()
  plt.close()
  
if __name__ == '__main__':
  # Generate training data from feature vectors
  train_data = load_csv_files(TRAIN_DIR)
  test_data = load_csv_files(TEST_DIR)
  
  input_shape = [train_data.shape[1]]
  
  # Build the vae
  vae = VAE(
    input_shape,
    latent_space_dim=LATENT_SPACE_DIM
  )
  vae.summary()
  vae.compile()
  print("...Compiled!")
  
  # Fit the model
  history = vae.train(
    X_train=train_data,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    validation_data=test_data
  )
  
  # Plot the loss
  plot_loss(
    history, 
    'calculate_reconstruction_loss', 
    '../../img/vae_reconstruction_loss.png'
  )
  
  plot_loss(
    history, 
    '_calculate_kl_loss', 
    '../../img/vae_kl_loss.png'
  )
  
  plot_loss(
    history, 
    'loss', 
    '../../img/vae_loss.png'
  )