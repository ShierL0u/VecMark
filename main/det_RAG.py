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
nltk.data.path.append('home/autodl-tmp/nltk_data')
from nltk.tokenize import sent_tokenize
from utils1 import *
import logging

def split_text_into_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        nltk.download('punkt', quiet=True)
        sentences = sent_tokenize(str(text))
        return sentences
    except:
        sentences = re.split(r'[.!?]+', str(text))
        return [s.strip() for s in sentences if s.strip()]
    
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
        QA_mapping_file=f"data/mapping/{dataset_name}_g{gamma}_QAlight.json"
    
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
            question = create_formatted_questions(dataset_name, df, df_id, field_name, answer_data)
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

def detect_watermark_RAG(mapping_data, dataset_name, dataset_path:str, sig: str, K1: str, K2: str, L: int, gamma: int) -> dict:
    """修正后的水印检测函数 - 只检测被水印化的句子"""
    from utils1 import (
        hash_sig, select_ka_ca, hash_value_key, hash_int, 
        load_model, create_watermarked_llm
    )
    
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
    
    # 加载模型
    logger.info("Loading detection model...")
    model, tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    
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

if __name__ == "__main__":
    # 测试修正后的检测算法
    dataset_name = "nfcorpus"
    if dataset_name == "nfcorpus":
        gamma = 37
    elif dataset_name == "winemag5k":
        gamma = 463
    elif dataset_name == "winemag50k":
        gamma = 587

    # question_model = "deepseek-r1:latest"
    # question_model = "qwen:14b"
    question_model = "light"
    dataset_path = f"data/watermarked_data/VectorMark_{dataset_name}_g{gamma}.csv"
    try:
        Q_mapping = question_generate_light(dataset_name, dataset_path, gamma, question_model, question_Opt=True)
        QA_mapping = answer_generate(dataset_name, gamma, 
                                     question_model=question_model,
                                    #  answer_type="VectorDatabase",
                                     answer_model="qwen:14b")
        # QA_mapping = Q_mapping
        print(f"Loaded dataset with {len(QA_mapping)} records")
        
        results = detect_watermark_RAG(
            mapping_data=QA_mapping,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            sig="watermark_test",
            K1="Key11",
            K2="Keyword2",
            L=16,
            gamma=gamma
        )
        
        # 保存结果
        results_path = f"./data/det_res/detection_RAG_g{gamma}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detection results saved to {results_path}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    