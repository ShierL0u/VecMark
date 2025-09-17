import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import pandas as pd
import torch
import re
import gc
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from utils1 import *
import hashlib
import jieba
from collections import defaultdict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import logging

class SimHashFilteredRetriever(BaseRetriever):
    """带 SimHash 去重的 Retriever"""

    base_retriever: BaseRetriever
    threshold: float = 0.7

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. 先拿原始 docs
        docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        # 2. 去重
        return filter_similar_documents(docs, self.threshold)

def split_text_into_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        nltk.download('punkt', quiet=True)
        sentences = sent_tokenize(str(text))
        return sentences
    except:
        sentences = re.split(r'[.!?]+', str(text))
        return [s.strip() for s in sentences if s.strip()]

def simhash(text, hash_bits=64, use_stopwords=False, stopwords_file=None):
    # 分词
    words = jieba.lcut(text)
    v = [0] * hash_bits  # 初始化向量
    
    # 加载停用词
    stopwords = stopwords.words('english')
    
    # 计算词频作为权重
    word_weights = defaultdict(int)
    for word in words:
        if word.strip() and (not use_stopwords or word not in stopwords):
            word_weights[word] += 1
    
    # 无有效特征时返回全0指纹
    if not word_weights:
        return '0' * hash_bits
    
    # 加权累加
    for word, weight in word_weights.items():
        hash_value = hashlib.md5(word.encode('utf-8')).hexdigest()
        binary_hash = bin(int(hash_value, 16))[2:].zfill(hash_bits)[:hash_bits]
        for i in range(hash_bits):
            bit = int(binary_hash[i])
            v[i] += weight if bit == 1 else -weight
    
    # 生成指纹
    return ''.join('1' if value > 0 else '0' for value in v)

def hamming_distance(hash1, hash2):
    """计算两个指纹的汉明距离"""
    if len(hash1) != len(hash2):
        raise ValueError("指纹长度不一致")
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def get_similarity(hash1, hash2):
    """计算两个指纹的相似度（重复率）"""
    if len(hash1) != len(hash2):
        raise ValueError("指纹长度不一致")
    distance = hamming_distance(hash1, hash2)
    return (len(hash1) - distance) / len(hash1) * 100

def is_similar(hash1, hash2, threshold=70.0):
    """判断两个指纹是否相似"""
    return get_similarity(hash1, hash2) >= threshold

def filter_similar_documents(documents: List[Document], threshold: float = 0.7) -> List[Document]:
    """过滤掉SimHash相似度超过阈值的文档"""
    if len(documents) <= 1:
        return documents
    
    # 计算所有文档的SimHash
    doc_hashes = [(doc, simhash(doc.page_content)) for doc in documents]
    
    # 过滤相似文档
    filtered = []
    for doc, doc_hash in doc_hashes:
        # 检查当前文档与已保留文档的相似度
        similar = False
        for filtered_doc, filtered_hash in filtered:
            similar = is_similar(doc_hash, filtered_hash, threshold)
            if similar:
                break
        if not similar:
            filtered.append((doc, doc_hash))
    
    # 只返回文档部分
    return [doc for doc, _ in filtered]

def split_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def create_RAGllm_chain_simhash(vector_db, RAG_model: str = "qwen:14b",
                                source_documents=False,
                                retrieval_k: int = 15,
                                simhash_threshold: float = 0.7):
    answer_prompt = create_answer_generate_prompt()
    base_retriever = vector_db.as_retriever(search_kwargs={"k": retrieval_k})

    filtered_retriever = SimHashFilteredRetriever(
        base_retriever=base_retriever,
        threshold=simhash_threshold
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(model_name=RAG_model),
        chain_type="stuff",
        retriever=filtered_retriever, 
        chain_type_kwargs={"prompt": answer_prompt},
        return_source_documents=source_documents
    )
    return rag_chain

def compute_ppl(
    text: str, model, tokenizer,              
    ctx_length: int = 2048,
    watch_bar: bool = False,
    device: str = "cuda"
) -> float:

    encodings = tokenizer(text, return_tensors="pt").to(device)
    seq_len = encodings.input_ids.size(1)

    max_length = ctx_length
    stride = ctx_length
    nlls = []
    prev_end_loc = 0

    process = range(0, seq_len, stride)
    if watch_bar:
        process = tqdm(process, desc="Compute ppl")
    for begin_loc in process:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            loss = model(input_ids, labels=target_ids).loss
        nlls.append(loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    return ppl

@torch.no_grad()
def compute_ppl_batch(texts: list[str],
                      model, tokenizer,
                      ctx_length: int = 2048,
                      stride: int = None,
                      device: str = "cuda",
                      batch_size: int = 8):
    '''
        找batch中最大的ppl
    '''
    stride = stride or ctx_length
    tokenizer.pad_token = tokenizer.eos_token

    # 1. 一维 token 流
    concat = []
    for t in texts:
        concat += tokenizer(t, add_special_tokens=False,
                          truncation=False, padding=False)['input_ids']
    concat = torch.tensor(concat, dtype=torch.long, device=device)
    seq_len = concat.size(0)

    nlls = []
    for begin in range(0, seq_len, stride * batch_size):
        windows, targets, attn_masks = [], [], []
        for i in range(batch_size):
            start = begin + i * stride
            if start >= seq_len:          # 已经越界
                break
            end = start + ctx_length
            chunk = concat[start:end]     # 可能短于 ctx_length
            actual_len = chunk.size(0)

            # 2. 不足就 pad 到 ctx_length
            pad_len = ctx_length - actual_len
            if pad_len > 0:
                chunk = torch.cat([chunk, torch.full((pad_len,), tokenizer.pad_token_id,
                                                     device=device)])
            windows.append(chunk)

            # 3. 构造 label 和 attention mask
            tgt = chunk.clone()
            tgt[:-stride] = -100          # 掩掉非 stride 部分
            if pad_len > 0:               # pad 位置也掩掉
                tgt[-pad_len:] = -100
            targets.append(tgt)
            attn_masks.append(torch.cat([torch.ones(actual_len, device=device),
                                         torch.zeros(pad_len, device=device)]))

        if not windows:
            break
        batch_input = torch.stack(windows)
        batch_tgt   = torch.stack(targets)
        batch_mask  = torch.stack(attn_masks)

        # 4. forward（带 attention_mask 避免 pad 参与计算）
        outputs = model(batch_input,
                        attention_mask=batch_mask,
                        labels=batch_tgt)
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = batch_tgt[..., 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1)).reshape(*shift_labels.shape)
        # 5. 只统计非 pad 部分
        denom = (shift_labels != -100).sum(1)
        nlls.append(loss.sum(1) / denom.clamp(min=1))

    ppl = torch.exp(torch.cat(nlls).mean()).item()
    with torch.no_grad():
        window_ppls = torch.exp(loss.sum(1) / denom.clamp(min=1)) 
    max_ppl = window_ppls.max().item()  

    return max_ppl
    # return ppl

def question_generate(dataset_name, dataset_path, gamma, question_model="deepseek-r1:latest", question_Opt:bool=False):
    # 根据rewrite_mapping文件生成问题对

    mapping_file=f"data/mapping/{dataset_name}_g{gamma}_rewrite_mapping.json"
    if question_model=="deepseek-r1:latest":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAdeepseekr1.json"
    elif question_model=="qwen:14b":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAqwen14b.json"
    
    df = pd.read_csv(dataset_path)
    Columns = df.columns.tolist()
    
    if os.path.exists(QA_mapping_file):
        with open(QA_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("question load successfully ...")
        return data
    
    print("question generate begin ...")

    question_json_schema = {
        "title": "question",
        "description": "question for the text.",
        "type": "object",
        "properties": {
            "Question": {
                "type": "string",
                "description": "Create an appropriate question.",
            },
        },
        "required": ["Question"],
    }

    question_prompt = create_question_generate_prompt()
    question_llm = get_ollama_llm(model_name=question_model).with_structured_output(question_json_schema)
    question_chain = question_prompt | question_llm

    rewrite_mapping = {}
    mapping_path = mapping_file
    with open(mapping_path, 'r', encoding='utf-8') as f:
        rewrite_mapping = json.load(f)
    
    # 如果启用问题优化，提前创建向量数据库
    if question_Opt:
        vector_db = create_RAGdb(dataset_path, dataset_name=dataset_name)

    count = 0
    invalid_count = 0    

    for key in tqdm(rewrite_mapping.keys(), desc="question generate"):
        for field_name in rewrite_mapping[key].keys():
            if field_name == "bit_position" or field_name == "bit_value":
                continue
                
            df_id = extract_idx(key)
            line_data = extract_dataline(df, df_id)
            answer_data = rewrite_mapping[key][field_name]['overall_rewritten_sentence']
            
            # 生成初始问题
            result = question_chain.invoke({
                "Entity": line_data, 
                # "Columns": Columns,
                "Text": answer_data
            })
            count += 1
            # 如果启用问题优化，检查检索结果
            if question_Opt:
                max_retries = 3  # 最大重试次数
                retry_count = 0
                question_valid = False
                
                while retry_count < max_retries and not question_valid:
                    # 检索相似文档
                    docs = vector_db.similarity_search(result['Question'], k=1)
                    
                    # 检查检索结果是否包含答案
                    if docs and answer_data in docs[0].page_content:
                        question_valid = True
                    else:
                        # 重新生成问题
                        print(f"retry {key} times: {retry_count + 1}/{max_retries})")
                        result = question_chain.invoke({
                            "Entity": line_data, 
                            # "Columns": Columns,
                            "Text": answer_data
                        })
                        retry_count += 1
                
                # 如果达到最大重试次数仍然无效，记录
                if not question_valid:
                    invalid_count += 1
            
            # 存储问题
            rewrite_mapping[key][field_name]['QA'] = result

    with open(QA_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(rewrite_mapping, f, indent=2, ensure_ascii=False)
    
    if question_Opt:
        invalid_rate = invalid_count / count * 100
        print(f"count : {count} | invalid_count : {invalid_count} | invalid_rate : {invalid_rate:.2f}%")
    print(f"QA Rewrite mapping saved to {QA_mapping_file} (question)")
    return rewrite_mapping

def question_generate_light(dataset_name, dataset_path, gamma, question_model="deepseek-r1:latest", question_Opt:bool=False):
    # 根据rewrite_mapping文件生成问题对

    mapping_file=f"data/mapping/{dataset_name}_g{gamma}_rewrite_mapping.json"
    if question_model=="deepseek-r1:latest":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAdeepseekr1.json"
    elif question_model=="qwen:14b":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAqwen14b.json"
    elif question_model=="light":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAlight_duplicate.json"
    
    df = pd.read_csv(dataset_path)
    Columns = df.columns.tolist()
    
    if os.path.exists(QA_mapping_file):
        with open(QA_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("question load successfully ...")
        return data
    
    print("question generate(light) begin ...")

    rewrite_mapping = {}
    mapping_path = mapping_file
    with open(mapping_path, 'r', encoding='utf-8') as f:
        rewrite_mapping = json.load(f)
    
    # 如果启用问题优化，提前创建向量数据库
    if question_Opt:
        vector_db = create_RAGdb(dataset_path, dataset_name=dataset_name)

    count = 0
    valid_count = 0

    for key in tqdm(rewrite_mapping.keys(), desc="question generate"):
        for field_name in rewrite_mapping[key].keys():
            if field_name == "bit_position" or field_name == "bit_value":
                continue
                
            df_id = extract_idx(key)
            # line_data = extract_dataline(df, df_id)
            answer_data = rewrite_mapping[key][field_name]['overall_rewritten_sentence']
            question = create_formatted_questions(answer_data)
            rewrite_mapping[key][field_name]['QA'] = {}
            rewrite_mapping[key][field_name]['QA']['Question'] = question

            if question_Opt:
                count += 1
                docs = vector_db.similarity_search(question, k=1)
                # 检查检索结果是否包含答案
                if docs and answer_data in docs[0].page_content:
                    valid_count += 1
    
    with open(QA_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(rewrite_mapping, f, indent=2, ensure_ascii=False)
        
    if question_Opt:
        valid_rate = valid_count / count * 100
        print(f"count : {count} | valid_count : {valid_count} | valid_rate : {valid_rate:.2f}%")
    print(f"QA Rewrite mapping saved to {QA_mapping_file} (question)")
    return rewrite_mapping
    

def answer_generate(dataset_name, gamma, question_model, answer_type="RAGllm", answer_model="qwen:14b"):
    # 根据问题对生成答案，保存在原QAmapping文件中

    dataset_path=f"data/watermarked_data/VectorMark_{dataset_name}_g{gamma}.csv"
    if question_model=="deepseek-r1:latest":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAdeepseekr1.json"
    elif question_model=="qwen:14b":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAqwen14b.json"
    elif question_model=="light":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAlight.json"

    with open(QA_mapping_file, 'r', encoding='utf-8') as f:
        QA_mapping = json.load(f)


    vector_db = create_RAGdb(dataset_path, dataset_name=dataset_name)
    answer_chain = create_RAGllm_chain(vector_db=vector_db, RAG_model=answer_model)

    for key in tqdm(QA_mapping.keys(), desc="answer generate"):
        for field_name in QA_mapping[key].keys():
            if field_name=="bit_position" or field_name=="bit_value":
                continue

            if answer_type == "VectorDatabase":
                docs = vector_db.similarity_search(QA_mapping[key][field_name]['QA']['Question'], k=1)  # 返回最相关的5个文档
                QA_mapping[key][field_name]['QA']['Answer'] = docs[0].page_content
            else:
                answer_input = f"QUESTION: {QA_mapping[key][field_name]['QA']['Question']}"
                result = answer_chain.invoke({"query": answer_input})
                QA_mapping[key][field_name]['QA']['Answer'] = result['result']

    with open(QA_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(QA_mapping, f, indent=2, ensure_ascii=False)

    return QA_mapping

def answer_generate_duplicate(dataset_name, gamma, 
                            question_model, answer_type="RAGllm", 
                            answer_model="qwen:14b", duplicate_threshold=0.7):
    # 根据问题对生成答案，保存在原QAmapping文件中

    dataset_path=f"data/watermarked_data/VectorMark_{dataset_name}_g{gamma}.csv"
    if question_model=="deepseek-r1:latest":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAdeepseekr1.json"
    elif question_model=="qwen:14b":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAqwen14b.json"
    elif question_model=="light":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAlight_duplicate.json"

    with open(QA_mapping_file, 'r', encoding='utf-8') as f:
        QA_mapping = json.load(f)

    vector_db = create_RAGdb(dataset_path, dataset_name=dataset_name)
    answer_chain = create_RAGllm_chain_simhash(
                vector_db=vector_db, RAG_model=answer_model,
                simhash_threshold=duplicate_threshold)

    for key in tqdm(QA_mapping.keys(), desc="answer generate"):
        for field_name in QA_mapping[key].keys():
            if field_name=="bit_position" or field_name=="bit_value":
                continue

            if answer_type == "VectorDatabase":
                docs = vector_db.similarity_search(QA_mapping[key][field_name]['QA']['Question'], k=1)  # 返回最相关的5个文档
                QA_mapping[key][field_name]['QA']['Answer'] = docs[0].page_content
            else:
                answer_input = f"QUESTION: {QA_mapping[key][field_name]['QA']['Question']}"
                result = answer_chain.invoke({"query": answer_input})
                QA_mapping[key][field_name]['QA']['Answer'] = result['result']

    with open(QA_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(QA_mapping, f, indent=2, ensure_ascii=False)

    return QA_mapping

def answer_ppl_fliter(dataset_name, 
        gamma, question_model, 
        answer_type="RAGllm", answer_model="qwen:14b", 
        ppl_model=None, ppl_tokenizer=None, ppl_threshold=0.0, ppl_rate=1.0, ppl_ctx_length=2048):
    # 根据问题对生成答案，保存在原QAmapping文件中

    QA_mapping_file = f"data/mapping/{dataset_name}_g{gamma}_QAlight_{ppl_rate}ppl.json"

    with open(QA_mapping_file, 'r', encoding='utf-8') as f:
        QA_mapping = json.load(f)

    total_answer = 0
    over_ppl_answer = 0

    for key in tqdm(QA_mapping.keys(), desc="answer generate"):
        for field_name in QA_mapping[key].keys():
            if field_name=="bit_position" or field_name=="bit_value":
                continue
            total_answer += 1
            ppl = compute_ppl_batch([QA_mapping[key][field_name]['QA']['Answer']],
                                    ppl_model,
                                    ppl_tokenizer, 
                                    ctx_length=ppl_ctx_length, 
                                    stride=ppl_ctx_length,
                                    device="cuda")
            if ppl > ppl_threshold:
                QA_mapping[key][field_name]['QA']['over_ppl'] = ppl
                over_ppl_answer += 1

    with open(QA_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(QA_mapping, f, indent=2, ensure_ascii=False)

    print(f"total_answer: {total_answer} | over_ppl_answer: {over_ppl_answer}")
    print(f"over_ppl_answer_rate: {over_ppl_answer/total_answer}")

    return QA_mapping


def answer_generate_ppl(dataset_name, 
        gamma, question_model, 
        answer_type="RAGllm", answer_model="qwen:14b", 
        ppl_model=None, ppl_tokenizer=None, ppl_threshold=0.0, ppl_rate=1.0, ppl_ctx_length=2048):
    # 根据问题对生成答案，保存在原QAmapping文件中

    dataset_path=f"data/watermarked_data/VectorMark_{dataset_name}_g{gamma}.csv"
    if question_model=="deepseek-r1:latest":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAdeepseekr1.json"
    elif question_model=="qwen:14b":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAqwen14b.json"
    elif question_model=="light":
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAlight.json"

    with open(QA_mapping_file, 'r', encoding='utf-8') as f:
        QA_mapping = json.load(f)

    # # 确保设备一致性 - 强制清理之前的GPU状态
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()

    total_answer = 0
    over_ppl_answer = 0
    vector_db = create_RAGdb(dataset_path, dataset_name=dataset_name)
    answer_chain = create_RAGllm_chain(vector_db=vector_db, RAG_model=answer_model)

    for key in tqdm(QA_mapping.keys(), desc="answer generate"):
        for field_name in QA_mapping[key].keys():
            if field_name=="bit_position" or field_name=="bit_value":
                continue

            if answer_type == "VectorDatabase":
                docs = vector_db.similarity_search(QA_mapping[key][field_name]['QA']['Question'], k=1)  # 返回最相关的5个文档
                QA_mapping[key][field_name]['QA']['Answer'] = docs[0].page_content
            else:
                total_answer += 1
                answer_input = f"QUESTION: {QA_mapping[key][field_name]['QA']['Question']}"
                result = answer_chain.invoke({"query": answer_input})
                ppl = compute_ppl_batch([result['result']],
                                         ppl_model,
                                         ppl_tokenizer, 
                                         ctx_length=ppl_ctx_length, 
                                         stride=ppl_ctx_length,
                                         device="cuda")
                if ppl <= ppl_threshold:
                    QA_mapping[key][field_name]['QA']['Answer'] = result['result']
                else:
                    QA_mapping[key][field_name]['QA']['Answer'] = result['result']
                    QA_mapping[key][field_name]['QA']['over_ppl'] = ppl
                    over_ppl_answer += 1

    QA_mapping_file = f"data/mapping/{dataset_name}_g{gamma}_QAlight_{ppl_rate}ppl.json"
    with open(QA_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(QA_mapping, f, indent=2, ensure_ascii=False)

    print(f"total_answer: {total_answer} | over_ppl_answer: {over_ppl_answer}")
    print(f"over_ppl_answer_rate: {over_ppl_answer/total_answer}")

    return QA_mapping


def find_watermarked_sentences_from_mapping(text_content, keywords):
    """根据关键词找到被水印化的句子"""
    if not text_content or not keywords:
        return []
    
    sentences = split_text_into_sentences(text_content)
    watermarked_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # 检查句子是否包含任何关键词
        contains_keyword = any(keyword.lower() in sentence_lower for keyword in keywords)
        if contains_keyword:
            watermarked_sentences.append(sentence)
    
    return watermarked_sentences

def detect_watermark_in_sentences(sentences, detector_0, detector_1):
    """检测句子列表中的水印, 返回累积的z_score"""
    if not sentences:
        return 0.0, 0.0
    
    total_score_0 = 0.0
    total_score_1 = 0.0
    
    for sentence in sentences:
        if sentence.strip():  # 确保句子不为空
            try:
                # 对每个句子进行检测
                res_0 = detector_0.detect_watermark(sentence)
                res_1 = detector_1.detect_watermark(sentence)
                
                # 累积z_score（或其他评分指标）
                score_0 = res_0.get('z_score', res_0.get('score', 0))
                score_1 = res_1.get('z_score', res_1.get('score', 0))
                
                total_score_0 += score_0
                total_score_1 += score_1
                
                print(f"Sentence: {sentence[:50]}...")
                print(f"Score_0: {score_0:.4f}, Score_1: {score_1:.4f}")
                
            except Exception as e:
                print(f"Error detecting watermark in sentence: {e}")
                continue
    
    return total_score_0, total_score_1

def get_keywords_from_mapping(rewrite_mapping, entity_idx, field_name):
    """从重写映射中获取关键词"""
    entity_key = f"entity_{entity_idx}"
    
    if entity_key in rewrite_mapping:
        entity_data = rewrite_mapping[entity_key]
        if field_name in entity_data:
            field_data = entity_data[field_name]
            return field_data.get('key_words', [])
    
    return []

def get_rewritten_text_from_mapping(rewrite_mapping, entity_idx, field_name):
    """从重写映射中获取重写后的完整文本"""
    entity_key = f"entity_{entity_idx}"
    
    if entity_key in rewrite_mapping:
        entity_data = rewrite_mapping[entity_key]
        if field_name in entity_data:
            field_data = entity_data[field_name]
            return field_data.get('overall_rewritten_sentence', None)
    
    return None

def get_answer_text_from_mapping(rewrite_mapping, entity_idx, field_name):
    """从重写映射中获取重写后的完整文本"""
    entity_key = f"entity_{entity_idx}"
    
    if entity_key in rewrite_mapping:
        entity_data = rewrite_mapping[entity_key]
        if field_name in entity_data:
            field_data = entity_data[field_name]
            return field_data["QA"]["Answer"]
    
    return None

def detect_watermark_in_text_field_corrected(text_content, keywords, detector_0, detector_1):
    """修正后的文本字段水印检测 - 只检测包含关键词的句子"""
    if not text_content or not keywords:
        return 0.0, 0.0
    
    # 根据关键词找到被水印化的句子
    watermarked_sentences = find_watermarked_sentences_from_mapping(text_content, keywords)
    
    if not watermarked_sentences:
        print("No watermarked sentences found based on keywords")
        return 0.0, 0.0
    
    print(f"Found {len(watermarked_sentences)} watermarked sentences")
    
    # 只对水印句子进行检测
    return detect_watermark_in_sentences(watermarked_sentences, detector_0, detector_1)

def detect_watermark_RAG(mapping_data, dataset_name, 
                         model, tokenizer,
                         dataset_path:str, sig: str, K1: str, K2: str, L: int, gamma: int) -> dict:
    """修正后的水印检测函数 - 只检测被水印化的句子"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    logger.info("Initializing RAG-based watermark detection...")
    
    # 加载重写映射
    rewrite_mapping = mapping_data
    df = pd.read_csv(dataset_path)
    
    wm_bits = hash_sig(sig, L)
    KA, A, CA_combos = select_ka_ca(df, dataset_name=dataset_name)
    candidate_attrs = CA_combos + KA
    
    logger.info(f"Original watermark bits: {wm_bits}")
    logger.info(f"Candidate attributes: {candidate_attrs}")
    
    # 简化的统计信息
    detection_stats = {
        'original_bits': wm_bits,
        'bit_stats': [
            {
                'embed_count': 0,      # 该位置被嵌入的次数
                'tend_to_0': 0,        # 倾向于0的次数
                'tend_to_1': 0,        # 倾向于1的次数
                'z_score_0': 0.0,      # 累积的z_score_0
                'z_score_1': 0.0       # 累积的z_score_1
            } for _ in range(L)
        ],
        'match_rate': 0.0
    }
    
    # 加载模型 - 确保设备一致性
    logger.info("Loading detection model...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 检测过程
    logger.info(f"Starting detection on {len(df)} records")
    progress_bar = tqdm(rewrite_mapping.keys(), desc="Detecting watermark RAG")
    
    for idx in progress_bar:
        entity_data = rewrite_mapping[idx]
        bit_position = entity_data["bit_position"]
        bit_value = entity_data["bit_value"]
        # 统计该位置被嵌入的次数
        detection_stats['bit_stats'][bit_position]['embed_count'] += 1
        OBJECT_fields = select_OBJECT_fields(dataset_name=dataset_name)

        # 创建检测器
        hash_key_0 = hash_int(K2, bit_position, 0, mod_num=11)
        hash_key_1 = hash_int(K2, bit_position, 1, mod_num=11)
        detector_0 = create_watermarked_llm(model, tokenizer, hash_key=hash_key_0)
        detector_1 = create_watermarked_llm(model, tokenizer, hash_key=hash_key_1)
        
        try:
            # 累积所有字段的z_score
            total_score_0 = 0.0
            total_score_1 = 0.0
            num_fliter = 0
            
            for field in OBJECT_fields:
                # 文本字段：使用修正后的检测方法
                keywords = get_keywords_from_mapping(rewrite_mapping, idx[7:], field)
                rewritten_text = get_answer_text_from_mapping(rewrite_mapping, idx[7:], field)
                
                if keywords and rewritten_text:
                    # 使用关键词定位水印句子并检测
                    score_0, score_1 = detect_watermark_in_text_field_corrected(
                        rewritten_text, keywords, detector_0, detector_1
                    )
                    
                    total_score_0 += score_0
                    total_score_1 += score_1
                    
                    logger.debug(f"Entity {idx}, Field {field}: score_0={score_0:.4f}, score_1={score_1:.4f}")
                
                # 累积到对应bit位置的统计中
                detection_stats['bit_stats'][bit_position]['z_score_0'] += total_score_0
                detection_stats['bit_stats'][bit_position]['z_score_1'] += total_score_1
            
            # 判断该entity倾向于哪个bit
            if total_score_0 > total_score_1:
                detection_stats['bit_stats'][bit_position]['tend_to_0'] += 1
            else:
                detection_stats['bit_stats'][bit_position]['tend_to_1'] += 1
                    
        finally:
            # 清理GPU内存
            del detector_0
            del detector_1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
                
    
    # 计算最终结果
    matched_bits = 0
    detectable_bits = 16
    
    for bp in range(L):
        stats = detection_stats['bit_stats'][bp]
        if stats['embed_count'] > 0:
            # 基于倾向次数判断该bit位置的最终值
            detected_bit = 0 if stats['tend_to_0'] > stats['tend_to_1'] else 1
            
            if detected_bit == wm_bits[bp]:
                matched_bits += 1
    
    # 更新匹配率
    detection_stats['match_rate'] = matched_bits / detectable_bits if detectable_bits > 0 else 0.0
    
    # 输出简化的结果
    logger.info("\n=== Simplified Watermark Detection Summary ===")
    logger.info(f"Original Watermark Bits: {wm_bits}")
    logger.info(f"Overall Match Rate: {detection_stats['match_rate']:.2%}")
    logger.info(f"Detectable Bits: {detectable_bits}/{L}")
    
    logger.info("\nBit-Level Statistics:")
    header = "Bit | Embed_Count | Tend_to_0 | Tend_to_1 | Z_Score_0 | Z_Score_1 | Original | Detected | Match"
    logger.info(header)
    logger.info("-" * len(header))
    
    for bp in range(L):
        stats = detection_stats['bit_stats'][bp]
        if stats['embed_count'] > 0:
            detected_bit = 0 if stats['tend_to_0'] > stats['tend_to_1'] else 1
            is_match = "✓" if detected_bit == wm_bits[bp] else "✗"
            
            log_entry = f"{bp:3d} | {stats['embed_count']:11d} | {stats['tend_to_0']:9d} | {stats['tend_to_1']:9d} | {stats['z_score_0']:9.2f} | {stats['z_score_1']:9.2f} | {wm_bits[bp]:8d} | {detected_bit:8d} | {is_match:5s}"
            logger.info(log_entry)
        else:
            log_entry = f"{bp:3d} | {stats['embed_count']:11d} | {stats['tend_to_0']:9d} | {stats['tend_to_1']:9d} | {stats['z_score_0']:9.2f} | {stats['z_score_1']:9.2f} | {wm_bits[bp]:8d} | {'N/A':8s} | {'N/A':5s}"
            logger.info(log_entry)
    
    return detection_stats

def ppl_distribution(ppl_path):
    """
    统计JSON文件中over_ppl字段的分布情况

    Args:
        ppl_path (str): 包含over_ppl数据的JSON文件路径

    Returns:
        dict: 包含统计信息的字典
    """
    import numpy as np
    from collections import Counter

    try:
        # 读取JSON文件
        with open(ppl_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有over_ppl值
        over_ppl_values = []
        for key in data.keys():
            if  'over_ppl' in data[key]['description']["QA"]:
                over_ppl_values.append(data[key]['description']["QA"]['over_ppl'])

        if not over_ppl_values:
            print("未找到任何over_ppl数据")
            return None

        # 计算基本统计信息
        stats = {
            'total_samples': len(over_ppl_values),
            'min_value': float(min(over_ppl_values)),
            'max_value': float(max(over_ppl_values)),
            'mean': float(np.mean(over_ppl_values)),
            'median': float(np.median(over_ppl_values)),
            'std_dev': float(np.std(over_ppl_values)),
            'q25': float(np.percentile(over_ppl_values, 25)),
            'q75': float(np.percentile(over_ppl_values, 75))
        }

        # 按区间统计分布
        bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, float('inf')]
        labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-200', '200-300', '300-400', '400+']

        hist, _ = np.histogram(over_ppl_values, bins=bins)
        distribution = {}

        for i, (label, count) in enumerate(zip(labels, hist)):
            percentage = (count / len(over_ppl_values)) * 100
            distribution[label] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }

        stats['distribution'] = distribution

        # 打印统计结果
        print("="*60)
        print("OVER_PPL 分布统计分析")
        print("="*60)
        print(f"总样本数: {stats['total_samples']}")
        print(f"最小值: {stats['min_value']:.4f}")
        print(f"最大值: {stats['max_value']:.4f}")
        print(f"平均值: {stats['mean']:.4f}")
        print(f"中位数: {stats['median']:.4f}")
        print(f"标准差: {stats['std_dev']:.4f}")
        print(f"25%分位数: {stats['q25']:.4f}")
        print(f"75%分位数: {stats['q75']:.4f}")
        print("\n分布区间统计:")
        print("-"*50)
        print(f"{'区间':<12} {'数量':<8} {'百分比':<8}")
        print("-"*50)

        for label, data in distribution.items():
            print(f"{label:<12} {data['count']:<8} {data['percentage']:.1f}%")

        print("="*60)

        return stats

    except FileNotFoundError:
        print(f"错误: 找不到文件 {ppl_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: JSON文件格式错误 {ppl_path}")
        return None
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

def test_compute_ppl_batch(text, ppl_ctx_length=2048):
    ppl_model, ppl_tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ppl = compute_ppl_batch(
        texts=[text],
        model=ppl_model,
        tokenizer=ppl_tokenizer,
        ctx_length=ppl_ctx_length,
        stride=ppl_ctx_length,
        device="cuda"
    )
    return ppl

def RAG_attack_ppl(dataset_name, ppl_rate, original_ppl=0, num_samples=-1, regenerate=False):
    if dataset_name == "nfcorpus":
        text_column = "text"
        gamma = 37
    elif dataset_name == "winemag5k":
        text_column = "description"
        gamma = 463
    elif dataset_name == "winemag50k":
        text_column = "description"
        gamma = 587

    # question_model = "deepseek-r1:latest"
    # question_model = "qwen:14b"
    question_model = "light"
    dataset_path = f"data/watermarked_data/VectorMark_{dataset_name}_g{gamma}.csv"
    original_dataset_path, original_dataset_name = to_original_path(dataset_path)
    original_dataset = pd.read_csv(original_dataset_path)
    if num_samples != -1:
        original_dataset = original_dataset.sample(num_samples)
    original_texts = original_dataset[text_column].astype(str).tolist()
    ppl_model, ppl_tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

    # 决定滑动窗口大小
    original_max_tokens = find_dataset_max_tokens(original_dataset, text_column)
    original_min_tokens = find_dataset_min_tokens(original_dataset, text_column)
    print(f"Original max tokens: {original_max_tokens}")
    print(f"Original min tokens: {original_min_tokens}")
    ppl_ctx_length = 2048
    if original_ppl == 0:
        ppls = []
        original_text_lists = split_list(original_texts, 100)
        for original_texts in tqdm(original_text_lists):
            ppl = compute_ppl_batch(
                texts=original_texts,
                model=ppl_model,
                tokenizer=ppl_tokenizer,
                ctx_length=ppl_ctx_length,
                stride=ppl_ctx_length,
                device="cuda"
            )
            ppls.append(ppl)
        original_ppl = max(ppls)
        percentiles = np.arange(10, 100, 10)  # [10, 20, ..., 90]
        ppl_percentiles = np.percentile(ppls, percentiles)
        for p, val in zip(percentiles, ppl_percentiles):
            print(f"{p}th percentile: {val:.4f}")
    ppl_threshold = original_ppl*ppl_rate

    print(f"Original PPL: {original_ppl}   |   PPL Threshold: {ppl_threshold}")

    try:
        if regenerate:
            Q_mapping = question_generate_light(dataset_name, dataset_path, gamma, question_model, question_Opt=True)
            QA_mapping = answer_generate_ppl(dataset_name, gamma, 
                                        question_model=question_model,
                                        #  answer_type="VectorDatabase",
                                        answer_model="qwen:14b",
                                        ppl_model=ppl_model,
                                        ppl_tokenizer=ppl_tokenizer,
                                        ppl_threshold=ppl_threshold,
                                        ppl_ctx_length=ppl_ctx_length,
                                        ppl_rate=ppl_rate)
        else:
            QA_mapping = answer_ppl_fliter(dataset_name, gamma, question_model, ppl_model=ppl_model, ppl_tokenizer=ppl_tokenizer, ppl_threshold=ppl_threshold, ppl_rate=ppl_rate)
        # QA_mapping = Q_mapping
        print(f"Done with {len(QA_mapping)} records")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

def RAG_attack_duplicate(dataset_name, duplicate_threshold=0.7, num_samples=-1):
    if dataset_name == "nfcorpus":
        text_column = "text"
        gamma = 37
    elif dataset_name == "winemag5k":
        text_column = "description"
        gamma = 463
    elif dataset_name == "winemag50k":
        text_column = "description"
        gamma = 587

    # question_model = "deepseek-r1:latest"
    # question_model = "qwen:14b"
    question_model = "light"
    dataset_path = f"data/watermarked_data/VectorMark_{dataset_name}_g{gamma}.csv"
    model, tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

    try:
        Q_mapping = question_generate_light(dataset_name, dataset_path, gamma, question_model, question_Opt=True)
        # QA_mapping = answer_generate_duplicate(dataset_name, gamma, 
        #                             question_model=question_model,
        #                             #  answer_type="VectorDatabase",
        #                             answer_model="qwen:14b",
        #                             duplicate_threshold=duplicate_threshold)
        with open(f"data/mapping/{dataset_name}_g{gamma}_QAlight_duplicate.json", 'r', encoding='utf-8') as f:
            QA_mapping = json.load(f)

        num_fliter = 0
        for key in QA_mapping.keys():
            if QA_mapping[key]['description']["QA"]['Answer'] == "No relevant information found.":
                num_fliter += 1

        results = detect_watermark_RAG(
            mapping_data=QA_mapping,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            model=model,
            tokenizer=tokenizer,
            sig="watermark_test",
            K1="Key11",
            K2="Keyword2",
            L=16,
            gamma=gamma
        )

        print(f"Done with {len(QA_mapping)} records")
        print(f"num_fliter: {num_fliter} | fliter_rate: {num_fliter/len(QA_mapping):.2%}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    
def test_ppl_distribution(ppl_path):
    stats = ppl_distribution(ppl_path)
    print(stats)

if __name__ == "__main__":
    # 测试修正后的检测算法
    # test_ppl_distribution(ppl_path="data/mapping/winemag50k_g587_QAlight_1.1ppl.json")

    # RAG_attack_ppl(dataset_name="winemag50k", original_ppl=100, ppl_rate=1.1, num_samples=-1)

    RAG_attack_duplicate(dataset_name="winemag50k", duplicate_threshold=0.7, num_samples=-1)

    # ppl = test_compute_ppl_batch(text="This is a cat.", ppl_ctx_length=2048)
    # print(ppl)
    


