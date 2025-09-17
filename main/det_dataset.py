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
nltk.data.path.append('~/autodl-tmp/nltk_data')
from nltk.tokenize import sent_tokenize
from utils1 import select_OBJECT_fields
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
    """检测句子列表中的水印，返回累积的z_score"""
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

def detect_watermark_v3(df: pd.DataFrame, dataset_name, mapping_path:str, sig: str, K1: str, K2: str, L: int, gamma: int, onlyTEXT:bool=False) -> dict:
    """修正后的水印检测函数 - 只检测被水印化的句子"""
    from utils1 import (
        hash_sig, select_ka_ca, hash_value_key, hash_int, 
        load_model, create_watermarked_llm
    )
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    logger.info("Initializing corrected watermark detection...")
    
    # 加载重写映射
    rewrite_mapping = {}
    mapping_path = mapping_path
    try:
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                rewrite_mapping = json.load(f)
        logger.info(f"Loaded {len(rewrite_mapping)} rewrite mappings")
    except Exception as e:
        logger.error(f"Error loading rewrite mapping: {e}")
        return {}
    
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
    progress_bar = tqdm(df.iterrows(), total=len(df), desc="Detecting watermark v3")
    
    for idx, row in progress_bar:
        try:
            PA = None
            BP = None
            
            # 定位阶段
            for I in candidate_attrs:
                val = str(row[I]) if isinstance(I, str) else "".join(str(row[f]) for f in I if f)
                if hash_value_key(val, K1, gamma) == 0:
                    PA = I
                    BP = hash_value_key(val, K2, L)
                    break
            
            if not PA:
                continue
            
            # 统计该位置被嵌入的次数
            detection_stats['bit_stats'][BP]['embed_count'] += 1
            WA_fields = list(set(A) - set([PA] if isinstance(PA, str) else PA))
            # OBJECT_fields = [field for field in WA_fields if field in df.select_dtypes(include=['object']).columns]
            OBJECT_fields = select_OBJECT_fields(dataset_name=dataset_name)

            # 创建检测器
            hash_key_0 = hash_int(K2, BP, 0, mod_num=11)
            hash_key_1 = hash_int(K2, BP, 1, mod_num=11)
            detector_0 = create_watermarked_llm(model, tokenizer, hash_key=hash_key_0)
            detector_1 = create_watermarked_llm(model, tokenizer, hash_key=hash_key_1)
            
            try:
                # 累积所有字段的z_score
                total_score_0 = 0.0
                total_score_1 = 0.0
                
                for field in WA_fields:
                    field_val = str(row[field])
                    
                    if field in OBJECT_fields:
                        # 文本字段：使用修正后的检测方法
                        keywords = get_keywords_from_mapping(rewrite_mapping, idx, field)
                        rewritten_text = get_rewritten_text_from_mapping(rewrite_mapping, idx, field)
                        
                        if keywords and rewritten_text:
                            # 使用关键词定位水印句子并检测
                            score_0, score_1 = detect_watermark_in_text_field_corrected(
                                rewritten_text, keywords, detector_0, detector_1
                            )
                            
                            total_score_0 += score_0
                            total_score_1 += score_1
                            
                            logger.debug(f"Entity {idx}, Field {field}: score_0={score_0:.4f}, score_1={score_1:.4f}")
                        
                    elif not onlyTEXT:
                        # 非文本字段：使用原有检测方法
                        res_0 = detector_0.detect_watermark(field_val)
                        res_1 = detector_1.detect_watermark(field_val)
                        
                        score_0 = res_0.get('z_score', res_0.get('score', 0))
                        score_1 = res_1.get('z_score', res_1.get('score', 0))
                        
                        total_score_0 += score_0
                        total_score_1 += score_1
                
                # 累积到对应bit位置的统计中
                detection_stats['bit_stats'][BP]['z_score_0'] += total_score_0
                detection_stats['bit_stats'][BP]['z_score_1'] += total_score_1
                
                # 判断该entity倾向于哪个bit
                if total_score_0 > total_score_1:
                    detection_stats['bit_stats'][BP]['tend_to_0'] += 1
                elif total_score_0 < total_score_1:
                    detection_stats['bit_stats'][BP]['tend_to_1'] += 1
                        
            finally:
                # 清理GPU内存
                del detector_0
                del detector_1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            continue
    
    # 计算最终结果
    matched_bits = 0
    detectable_bits = 0
    
    for bp in range(L):
        stats = detection_stats['bit_stats'][bp]
        if stats['embed_count'] > 0:
            detectable_bits += 1
            # 基于倾向次数判断该bit位置的最终值
            if stats['tend_to_0'] > stats['tend_to_1']:
                detected_bit = 0
            elif stats['tend_to_0'] < stats['tend_to_1']:
                detected_bit = 1
            else:
                detected_bit = -1
            
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
            if stats['tend_to_0'] > stats['tend_to_1']:
                detected_bit = 0
            elif stats['tend_to_0'] < stats['tend_to_1']:
                detected_bit = 1
            else:
                detected_bit = -1
            is_match = "✓" if detected_bit == wm_bits[bp] else "✗"
            
            log_entry = f"{bp:3d} | {stats['embed_count']:11d} | {stats['tend_to_0']:9d} | {stats['tend_to_1']:9d} | {stats['z_score_0']:9.2f} | {stats['z_score_1']:9.2f} | {wm_bits[bp]:8d} | {detected_bit:8d} | {is_match:5s}"
            logger.info(log_entry)
        else:
            log_entry = f"{bp:3d} | {stats['embed_count']:11d} | {stats['tend_to_0']:9d} | {stats['tend_to_1']:9d} | {stats['z_score_0']:9.2f} | {stats['z_score_1']:9.2f} | {wm_bits[bp]:8d} | {'N/A':8s} | {'N/A':5s}"
            logger.info(log_entry)
    
    return detection_stats

if __name__ == "__main__":
    # 测试修正后的检测算法'
    dataset_name = "clean_winemag50k"
    if dataset_name == "nfcorpus":
        gamma = 37
        df = pd.read_csv("/root/autodl-tmp/dataset/nfcorpus_corpus.csv")
    elif dataset_name == "winemag5k":
        gamma = 271
        df = pd.read_csv("/root/autodl-tmp/dataset/winemag_sub_dataset_5k.csv")
    elif dataset_name == "winemag50k":
        gamma = 587
        df = pd.read_csv("/root/autodl-tmp/dataset/winemag_sub_dataset_50k.csv")
    elif dataset_name == "clean_winemag50k":
        gamma = 587
        df = pd.read_csv("/root/autodl-tmp/dataset/winemag_sub_dataset_50k.csv")
    elif dataset_name == "FCT100k":
        gamma = 1721
        df = pd.read_csv("data/parquet/watermarked_FCT100k_g1721.csv")
    try:
        print(f"Loaded dataset with {len(df)} records")
        
        results = detect_watermark_v3(
            df=df,
            dataset_name=dataset_name,
            mapping_path=f"data/mapping/{dataset_name}_g{gamma}_rewrite_mapping.json",
            sig="watermark_test",
            K1="Key11",
            K2="Keyword2",
            L=16,
            gamma=gamma,
            # onlyTEXT=True
        )
        
        # 保存结果
        # results_path = f"./data/det_res/detection_results_{dataset_name}_g{gamma}.json"
        # with open(results_path, 'w', encoding='utf-8') as f:
        #     json.dump(results, f, indent=2, ensure_ascii=False)
        # print(f"Detection results saved to {results_path}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise