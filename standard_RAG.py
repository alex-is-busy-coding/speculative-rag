import os
import sys
import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from config import HF_token

from sentence_transformers import SentenceTransformer
import faiss
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
)

from bioasq_fetcher import BioASQDataFetcher


class Config:
    # Retrieval
    # EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 10
    EMBED_BATCH = 512

    # LLM
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    MAX_NEW_TOKENS = 256
    os.environ['HF_TOKEN'] = HF_token

    # Evaluation
    TEST_SAMPLES = 100


def print_environment():
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version  : {torch.version.cuda}")
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  FAISS   : {faiss.__version__}")
    print(f"  Embedding : {Config.EMBEDDING_MODEL}")
    print(f"  LLM  : {Config.LLM_MODEL}")
    print(f"  Top-K : {Config.TOP_K}")
    print(f"  Max new tokens: {Config.MAX_NEW_TOKENS}")
    print(f"  Eval samples  : {Config.EVAL_SAMPLES}")
    print("=" * 60)

    

class DocumentStore:
    '''
    Encodes a corpus of passages with a SentenceTransformer model and stores
    them in a FAISS flat inner-product index
    '''
    def __init__(self, model_name = Config.EMBEDDING_MODEL):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'DocumentStore loading {model_name} on {device}')
        self.encoder = SentenceTransformer(model_name, device = device)
        self.index = None
        self.passages = None

    # Uncomment if using BGE embedding model
    # @staticmethod
    # def query_text(q):
    #     return f'Represent this sentence for searching relevant passages: {q}'

    def build_index(self,passages):
        print(f'DocumentStore Encoding {len(passages)} passages...')
        self.passages = passages
        embeddings = self.encoder.encode(
            passages,
            batch_size = Config.EMBED_BATCH,
            # show_progress = True,
            convert_to_numpy = True,
            normalize_embeddings = True,
        ).astype(np.float32)
    
        dim = embeddings.shape[1]
    
        # Use GPU index when CUDA is available
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            flat_index = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, flat_index)
            print('DocumentStore using GPU FAISS index')
    
        else:
            self.index = faiss.IndexFlatIP(dim)
            print('DocumentStore using CPU FAISS index')
    
        self.index.add(embeddings)
        print(f'DocumentStore Index ready {self.index.ntotal} vectors, dim = {dim}')


    def retrieve(self, query, top_k = Config.TOP_K):
        'Returns [(passage_text, score), ...] sorted by descending similarity.'
        if self.index is None:
            raise RuntimeError('Call build_index() before retrieve()')
        q_emb = self.encoder.encode(
            [query],
            # [query_text(query)], # when using BGE model
            convert_to_numpy = True,
            normalize_embeddings = True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, top_k)

        return [
            self.passages[idx]
            for s, idx in zip(scores[0], indices[0])
            if idx != -1
        ]


class LLMGenerator:
    def __init__(
        self,
        model_name = Config.LLM_MODEL,
        max_new_tokens = Config.MAX_NEW_TOKENS
    ):
        self.max_new_tokens = max_new_tokens,

        print(f"LLMGenerator Loading '{model_name}'...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = token)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto",             
        #     token=token,
        # )

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = 'nf4',
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_cfg,
            device_map = 'auto',
            token = token
        )
        self.model.eval()

        print('LLMGenerator Model ready')


    def generate(self, prompt):
        'Returns only the newly generated text.'
        inputs = self.tokenizer(prompt, return_tensors = 'pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample = False,
                pad_token_id = self.tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens = True).strip()


class StandardRAG:
    '''
    Retrieve top_k documents
    Concatenate all of them into one prompt
    Call LLM to get the final answer
    '''
    

    def __init__(self, doc_store, llm, top_k = Config.TOP_K):
        self.doc_store = doc_store
        self.llm = llm
        self.top_k = top_k

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def build_prompt(self, question, docs):
        PROMPT_TEMPLATE = (
            '[INST] You are a retrieval-augmented assistant.'
            'Below is an instruction that describes a task.'
            "Write a response for it and state your explanation supporting your response.\n\n"
            'Evidence : {evidence}\n\n'
            'Instruction : {question}\n\n'
            '[/INST] The response is: '
        
    )
        evidence = '\n'.join(f"[{i}] {text}" for i, text in enumerate(docs, 1))
        return PROMPT_TEMPLATE.format(evidence = evidence, question = question)

    def answer(self, question, verbose = True):
        '''
        Returns:
        - question : original question
        - retrieved_docs : list of (passage, score)
        - answer : generated answer
        - latency_s : wall-clock seconds
        '''
        t0 = self.sync()

        # retrieve related knowledge based on  prompt
        retrieved = self.doc_store.retrieve(question, top_k = self.top_k)
        

        # build prompt with retrieved knowledge
        prompt = self.build_prompt(question, retrieved)

        if verbose:
            print('\n'+'='*60)
            print(f'Q : {question}')
            print('\n'+'-'* 50)
            print('Retrieved passages: ')
            for i, doc in enumerate(retrieved[:3], 1):
                print(f"  [{i}] {doc[:400]}{'...' if len(doc) > 400 else ''}\n")

        # generate answers with built prompt
        answer_text = self.llm.generate(prompt)
        latency = self.sync() - t0

        if verbose:
            ans = answer_text[:600] + ('...' if len(answer_text) > 600 else '')
            print('-'* 50)
            print(f'A : {ans}')
            print('\n'+'-'* 50)
            print(f'Latency : {latency} s')

        return {
            'question' : question,
            'retrieved_docs': retrieved,
            'answer' : answer_text,
            'latency_s' : latency,
        }
    def batch_answer(self, questions):
        return [
            self.answer(q, verbose = False)
            for q in tqdm(questions, desc = '        Standard RAG')
        ]




def main():
    print('=' * 60)
    print_environment()
    print('=' * 60)

    # Load dataset
    fetcher = BioASQDataFetcher()
    fetcher.load_data()

    passages = fetcher.get_all_passages()
    questions, answers = fetcher.get_qa_pairs()

    fetcher.inspect()

    n = Config.TEST_SAMPLES
    if n is not None:
        questions = questions[:n]
        gold_answers = answers[:n]

    print(f"Corpus :  {len(passages)} passages")
    print(f"Queries : {len(questions)} questions\n")

    

    # Build FAISS index
    doc_store = DocumentStore()
    doc_store.build_index(passages)

    # Load LLM Model
    llm = LLMGenerator()

    # Build pipeline
    rag = StandardRAG(doc_store = doc_store, llm = llm)

    # Single query demo
    print('\n---Single query demo----')
    demo = rag.answer(questions[0], verbose = True)
    print(f'Gold : {gold_answers[0]}\n')

    
    rag.batch_answer(questions[1:4])

    

 







        



        



        
        
        
        
        
        
    
    

    