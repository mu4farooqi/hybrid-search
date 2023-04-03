import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)

def compute_dense_embeddings(image_batch):
  # Gives 512 dimensional dense vector for images.
  images = list(map(lambda path: Image.open(path), image_batch))
  return model.encode(images)

