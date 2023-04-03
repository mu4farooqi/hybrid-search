import torch
import orjson as json
from fastapi import FastAPI, Request
from fastapi.responses import Response
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(1)


tokenizer = AutoTokenizer.from_pretrained(
  'naver/splade-cocondenser-ensembledistil')
model = Splade(
        'naver/splade-cocondenser-ensembledistil', agg="max").to(DEVICE)


app = FastAPI()


def json_response(content, status_code=200):
  option = json.OPT_SERIALIZE_NUMPY | json.OPT_NAIVE_UTC
  return Response(content=json.dumps(content, option=option), status_code=status_code)


@app.on_event("startup")
async def startup():
  # Warmup the model so that first call doesn't get stuck.
  _ = get_embedding(['It is 5 a.m. here, so I apologize. I might not be as exciting as I usually am. What Id like to do is spend just about 20 minutes talking to you all.'])


def get_embedding(text_batch):
  inputs = tokenizer(text_batch, max_length=512, padding=True,
                     truncation=True, return_tensors='pt').to(DEVICE)
  with torch.no_grad():
    sparse_embessings = model(d_kwargs=inputs)["d_rep"]

  cols = [embedding.nonzero().squeeze().cpu().tolist()
          for embedding in sparse_embessings]
  weights = [embedding[cols[idx]].cpu().tolist()
             for idx, embedding in enumerate(sparse_embessings)]
  return [{f'feature_{c}': w for c, w in zip(col, weight)} for col, weight in zip(cols, weights)]


@app.post('/sparse')
async def sparse(request: Request):
  data = await request.json()
  embeddings = get_embedding(data['docs'])

  return json_response(content={'embeddings': embeddings})