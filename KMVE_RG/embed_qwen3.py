from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn.functional as F
import faiss
import json
from transformers import AutoTokenizer, AutoModel
import os

INDEX_DIR = "/home/chenzhw/ultrasound_report_gen/US-Report-Gen/qwen3_embed"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

app = FastAPI()

# 加载模型
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', trust_remote_code=True, padding_side='left')
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B', trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

# 全局向量库和文段
index = None
doc_texts = []
doc_uids = [] 

def load_index_and_meta():
    global index, doc_texts, doc_uids

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            doc_texts = meta["doc_texts"]
            doc_uids = meta["doc_uids"]
        print(f"[INFO] Loaded index and metadata from {INDEX_DIR}")
        return True
    return False

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def embed_texts(texts: List[str]) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors='pt')
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()

# 请求体：从JSON文件读取
class InitIndexFromFileRequest(BaseModel):
    json_path: str




@app.post("/init_index")
def init_index_from_json(request: InitIndexFromFileRequest):
    global index, doc_texts, doc_uids

    # 尝试加载已有索引
    if load_index_and_meta():
        return {"status": "loaded_from_file", "doc_count": len(doc_texts)}

    try:
        with open(request.json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        doc_texts = []
        doc_uids = []
        for split_items in data.values():
            for item in split_items:
                if 'finding' in item:
                    doc_texts.append(item['finding'])
                    doc_uids.append(item['uid'])

        if not doc_texts:
            return {"error": "No findings found in JSON file."}

        os.makedirs(INDEX_DIR, exist_ok=True)

        index = None
        batch_size = 100

        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i + batch_size]
            batch_embeddings = embed_texts(batch_texts)
            batch_embeddings = batch_embeddings.cpu().numpy()

            if index is None:
                dim = batch_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)

            index.add(batch_embeddings)
            torch.cuda.empty_cache()
            print(f"Added batch {i//batch_size + 1}: size {len(batch_texts)}")

        # 保存索引
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "doc_texts": doc_texts,
                "doc_uids": doc_uids
            }, f, ensure_ascii=False, indent=2)

        return {"status": "index_built", "doc_count": len(doc_texts)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# 查询接口保持不变
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query")
def query(request: QueryRequest):
    global index, doc_texts
    if index is None:
        return {"error": "index not initialized"}

    query_embed = embed_texts([request.query]).numpy()
    scores, indices = index.search(query_embed, request.top_k)

    results = [
        {"text": doc_texts[i], "score": float(scores[0][j])}
        for j, i in enumerate(indices[0])
    ]
    return {"results": results}

@app.post("/search")
def search_similar_findings(request: QueryRequest):
    global index, doc_texts, doc_uids

    if index is None or not doc_texts:
        return {"error": "Index is not initialized."}

    # 获取 query 的 embedding
    query_embedding = embed_texts([request.query]).cpu().numpy()

    # FAISS 查询
    D, I = index.search(query_embedding, request.top_k)

    # 组装返回结果
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(doc_texts):
            results.append({
                "uid": doc_uids[idx],
                "finding": doc_texts[idx],
                "score": float(score)
            })

    return {"results": results}