from openai.embeddings_utils import get_embeddings


def compute_dense_embeddings(text_batch):
  # Gives 1536 dimensional dense vector for text.

  embeddings = []
  for i in range(0, len(text_list), 2048):
    embeddings.extend(get_embeddings(text_list[i:i + 2048], 'text-embedding-ada-002'))

  return embeddings