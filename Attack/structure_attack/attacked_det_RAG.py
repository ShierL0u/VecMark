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
import matplotlib.pyplot as plt
import numpy as np

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

    for key in tqdm(rewrite_mapping.keys(), desc=f"{dataset_name} question generate"):
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

def question_generate_light(watermark_df:pd.DataFrame, rewrite_mapping:dict, 
                            dataset_name:str, column_name:str="description", 
                            question_model="deepseek-r1:latest")->dict:

    print("question generate(light) begin ...")

    question_mapping = {}

    for key in tqdm(rewrite_mapping.keys(), desc=f"{dataset_name} question generate"):
        watermark_data =rewrite_mapping[key][column_name]["overall_rewritten_sentence"]
        idx = extract_idx(key)
        if (watermark_df['id'] == idx).any():
            entity_data = rewrite_mapping[key]
            question_mapping[key] = entity_data
            question = create_formatted_questions(watermark_data)
            rewrite_mapping[key][column_name]['QA'] = {}
            rewrite_mapping[key][column_name]['QA']['Question'] = question
            # question_mapping[key] = {}
            # question_mapping[key]['bit_position'] = entity_data["bit_position"]
            # question_mapping[key]['bit_value'] = entity_data["bit_value"]
            # question = create_formatted_questions(watermark_data)
            # question_mapping[key]['watermared_data'] = watermark_data
            # question_mapping[key]['Question'] = question
        
    print(f"question generate(light) completed")
    return question_mapping
    

def answer_generate(watermarked_data_path, question_mapping:dict, dataset_name,
                    column_name:str="description", answer_type="RAGllm", answer_model="qwen3:8b")->dict:

    if os.path.exists(f"data/mapping/attacked_onRAG/{dataset_name}.json"):
        with open(f"data/mapping/attacked_onRAG/{dataset_name}.json", 'r', encoding='utf-8') as f:
            question_mapping = json.load(f)
        print("question mapping load successfully ...")
        return question_mapping

    vector_db = create_RAGdb(watermarked_data_path, dataset_name=dataset_name)
    answer_chain = create_RAGllm_chain(vector_db=vector_db, RAG_model=answer_model)

    # for key in tqdm(question_mapping.keys(), desc="answer generate"):
    for key in question_mapping.keys():

        if answer_type == "VectorDatabase":
            docs = vector_db.similarity_search(question_mapping[key][column_name]['QA']['Question'], k=1) 
            question_mapping[key][column_name]['QA']['Answer'] = docs[0].page_content
        else:
            answer_input = f"QUESTION: {question_mapping[key][column_name]['QA']['Question']}"
            result = answer_chain.invoke({"query": answer_input})
            question_mapping[key][column_name]['QA']['Answer'] = result['result']

    with open(f"data/mapping/attacked_onRAG/{dataset_name}.json", 'w', encoding='utf-8') as f:
        json.dump(question_mapping, f, indent=2, ensure_ascii=False)
    print("answer mapping saved to data/mapping/attacked_onRAG/{dataset_name}.json")

    return question_mapping


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
        # print("No watermarked sentences found based on keywords")
        return 0.0, 0.0
    
    # 只对水印句子进行检测
    return detect_watermark_in_sentences(watermarked_sentences, detector_0, detector_1)

def detect_watermark_RAG(QA_mapping, watermarked_data:pd.DataFrame, dataset_name, sig: str, K1: str, K2: str, L: int, gamma: int, model=None, tokenizer=None) -> dict:
    """修正后的水印检测函数 - 只检测被水印化的句子"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    logger.info("Initializing RAG-based watermark detection...")
    
    # 加载重写映射
    rewrite_mapping = QA_mapping
    
    wm_bits = hash_sig(sig, L)
    KA, A, CA_combos = select_ka_ca(watermarked_data, dataset_name=dataset_name)
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
    
    # 加载模型（如果未提供）
    if model is None or tokenizer is None:
        logger.info("Loading detection model...")
        model, tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    else:
        logger.info("Using provided model and tokenizer...")
    
    # 检测过程
    logger.info(f"Starting detection on {len(watermarked_data)} records")
    # progress_bar = tqdm(rewrite_mapping.keys(), desc=f"{dataset_name} Detecting watermark RAG")
    progress_bar = rewrite_mapping.keys()
    
    for idx in progress_bar:
        entity_data = rewrite_mapping[idx]
        bit_position = entity_data["bit_position"]
        bit_value = entity_data["bit_value"]
        # 统计该位置被嵌入的次数
        detection_stats['bit_stats'][bit_position]['embed_count'] += 1
        # OBJECT_fields = select_OBJECT_fields(dataset_name=dataset_name)
        OBJECT_fields = ['description']

        # 创建检测器
        hash_key_0 = hash_int(K2, bit_position, 0, mod_num=11)
        hash_key_1 = hash_int(K2, bit_position, 1, mod_num=11)
        detector_0 = create_watermarked_llm(model, tokenizer, hash_key=hash_key_0)
        detector_1 = create_watermarked_llm(model, tokenizer, hash_key=hash_key_1)
        
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
            
            # 累积到对应bit位置的统计中
            detection_stats['bit_stats'][bit_position]['z_score_0'] += total_score_0
            detection_stats['bit_stats'][bit_position]['z_score_1'] += total_score_1
        
        # 判断该entity倾向于哪个bit
        if total_score_0 > total_score_1:
            detection_stats['bit_stats'][bit_position]['tend_to_0'] += 1
        elif total_score_0 < total_score_1:
            detection_stats['bit_stats'][bit_position]['tend_to_1'] += 1
                
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
    logger.info(f"\n====================== {dataset_name} Detection Summary ============================")
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

def attack(df, base_df, attack_type, attack_rate):
    """
    对数据集进行攻击操作
    
    Args:
        df: 目标数据框
        base_df: 基础数据框，用于提供替换数据
        attack_type: 攻击类型
        attack_rate: 攻击比例 (0-1之间)
    
    Returns:
        攻击后的dataframe
    """
    df = df.copy()  # 创建副本避免修改原始数据
    
    if attack_type == "Deleted Entities":
        # 删除一定比例的行
        num_to_delete = int(len(df) * attack_rate)
        indices_to_delete = df.sample(n=num_to_delete).index
        df = df.drop(indices_to_delete)
        
    elif attack_type == "Inserted Entities":
        # 插入来自base_df的随机行
        num_to_insert = int(len(df) * attack_rate)
        rows_to_insert = base_df.sample(n=num_to_insert)
        df = pd.concat([df, rows_to_insert], ignore_index=True)
        
    elif attack_type == "Modified Entities":
        # 将df中的一定比例的行用base_df中随机行进行替换
        num_to_modify = int(len(df) * attack_rate)
        if num_to_modify > 0:
            # 随机选择要修改的行索引
            indices_to_modify = df.sample(n=num_to_modify).index
            # 从base_df中随机选择相同数量的行进行替换
            replacement_rows = base_df.sample(n=num_to_modify)
            # 重置replacement_rows的索引以匹配要替换的位置
            replacement_rows.index = indices_to_modify
            # 替换选中的行
            df.loc[indices_to_modify] = replacement_rows
        
    elif attack_type == "Modified Attributes":
        # 将df中的部分字段用来自base_df的相同列的其它字段值替换
        # 计算要修改的字段总数（行数 × 列数 × 攻击比例）
        total_fields = len(df) * len(df.columns)
        num_fields_to_modify = int(total_fields * attack_rate)
        
        if num_fields_to_modify > 0:
            # 随机选择要修改的字段（行和列的组合）
            all_field_positions = []
            for row_idx in df.index:
                for col_name in df.columns:
                    if col_name in base_df.columns:  # 确保base_df中也有这个列
                        all_field_positions.append((row_idx, col_name))
            
            # 随机选择要修改的字段位置
            if len(all_field_positions) > 0:
                fields_to_modify = pd.Series(all_field_positions).sample(
                    n=min(num_fields_to_modify, len(all_field_positions))
                ).tolist()
                
                for row_idx, col_name in fields_to_modify:
                    # 从base_df的相同列中随机选择一个值进行替换
                    random_value = base_df[col_name].sample(n=1).iloc[0]
                    df.at[row_idx, col_name] = random_value
    
    return df

def test_attack_effects_RAG():
    """
    测试不同攻击类型和攻击率下的RAG水印检出情况
    """
    # 创建保存结果的目录
    result_dir = "data/attacked_eva/detRAG_result"
    fig_dir = "data/attacked_eva/fig_onRAG"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # 加载数据
    print("Loading datasets...")
    base_df = pd.read_csv("/root/autodl-tmp/dataset/winemag-data-130k-v2.csv")
    df = pd.read_csv("/root/autodl-tmp/dataset/watermark_dataset/VectorMark_winemag50k_g587.csv")
    
    # 定义攻击类型
    attack_types = ["Deleted Entities", "Inserted Entities", "Modified Entities", "Modified Attributes"]
    
    # 定义攻击率
    attack_rates_high = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    attack_rates_low = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    
    # 存储所有结果
    all_results = {}
    
    # 创建所有测试组合
    all_test_combinations = []
    # 前三种攻击方式测试高攻击率
    for attack_type in attack_types[:-1]:
        for rate in attack_rates_high:
            all_test_combinations.append((attack_type, rate))
    
    # Modified Attributes测试低攻击率
    all_test_combinations.extend([("Modified Attributes", rate) for rate in attack_rates_low])
    
    print(f"Starting RAG attack effect testing, total {len(all_test_combinations)} test combinations...")
    
    # 使用tqdm显示总体进度
    overall_progress = tqdm(all_test_combinations, desc="Overall Progress", position=0)

    # 只加载一次模型
    model, tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    if os.path.exists(f"../dataset/watermark_dataset/VectorMark_winemag50k_g587_rewrite_mapping.json"):
        with open(f"../dataset/watermark_dataset/VectorMark_winemag50k_g587_rewrite_mapping.json", 'r', encoding='utf-8') as f:
            rewrite_mapping = json.load(f)
    else:
        rewrite_mapping = None
    
    for attack_type in attack_types:
        attack_results = {}
        
        # 根据攻击类型选择攻击率范围
        if attack_type == "Modified Attributes":
            # Modified Attributes使用低攻击率
            rates_to_test = attack_rates_low
        else:
            # 前三种攻击方式使用高攻击率
            rates_to_test = attack_rates_high
        
        for rate in rates_to_test:
            # 执行攻击
            attacked_df = attack(df, base_df, attack_type, rate)
            
            # 生成攻击后的数据集名称（驼峰命名法）
            attack_name_camel = attack_type.replace(' ', '')
            dataset_name = f"winemag50k_{attack_name_camel}_{rate}"
            
            
            question_mapping = question_generate_light(
                watermark_df=attacked_df,
                rewrite_mapping=rewrite_mapping,
                dataset_name=dataset_name,
                question_model="light", 
            )

            watermarked_data_path = f"../dataset/attacked_dataset/watermarked_{dataset_name}.csv"
            attacked_df.to_csv(watermarked_data_path, index=False)
            
            # 生成答案
            QA_mapping = answer_generate(
                watermarked_data_path=watermarked_data_path,
                question_mapping=question_mapping,
                dataset_name=dataset_name,
                answer_type="RAGllm", 
                answer_model="qwen:14b"
            )
            
            # 检测水印
            results = detect_watermark_RAG(
                QA_mapping=QA_mapping,
                watermarked_data=attacked_df,
                dataset_name=dataset_name,
                sig="watermark_test",
                K1="Key11",
                K2="Keyword2",
                L=16,
                gamma=587,
                model=model,
                tokenizer=tokenizer
            )
            
            # 检测完成后立即清理GPU资源
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # 保存结果
            match_rate = results.get('match_rate', 0.0)
            attack_results[rate] = {
                'match_rate': match_rate,
                'detection_stats': results
            }
            
            # 保存详细结果到文件
            result_filename = f"{attack_type.replace(' ', '_')}_{rate}.json"
            result_path = os.path.join(result_dir, result_filename)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 更新进度条描述
            overall_progress.set_description(f"{attack_type} - {rate} (Detection Rate: {match_rate:.4f})")
            overall_progress.update(1)
        
        all_results[attack_type] = attack_results
    
    overall_progress.close()
    
    # 绘制曲线图 - 为每种攻击类型分别绘制
    print("\nDrawing RAG detection rate curves...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 为每种攻击类型分别绘制全范围曲线图
    for attack_type in attack_types:
        if not all_results[attack_type]:
            print(f"No results found for {attack_type}, skipping...")
            continue

        plt.figure(figsize=(12, 8))

        # 收集数据
        if attack_type == "Modified Attributes":
            rates_to_plot = attack_rates_low
        else:
            rates_to_plot = attack_rates_high

        rates = sorted([rate for rate in rates_to_plot if rate in all_results[attack_type]])
        match_rates = [all_results[attack_type][rate]['match_rate'] for rate in rates]

        # 绘制曲线
        plt.plot(rates, match_rates,
                color='blue', marker='o',
                linewidth=2, markersize=6,
                label=attack_type)

        # 横坐标和纵坐标设置
        y_min, y_max = min(match_rates), max(match_rates)

        # 除了Modified Attributes外，横坐标设置为0到1的范围
        if attack_type != "Modified Attributes":
            plt.xlim(0, 1)
            # 设置横轴刻度为实际数据点
            plt.xticks(rates)
        else:
            # Modified Attributes使用实际数据范围
            x_min, x_max = min(rates), max(rates)
            plt.xlim(x_min, x_max)

        plt.ylim(0, 1.2)

        plt.xlabel('Attack Rate', fontsize=12)
        plt.ylabel('WER(%)', fontsize=12)
        plt.title(f'{attack_type} - Watermark Extraction Rate vs Attack Rate', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='lower left')
        plt.grid(True, alpha=0.3)

        # 保存图像
        safe_name = attack_type.replace(' ', '_')
        fig_path = os.path.join(fig_dir, f'{safe_name}_detection_curve.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {attack_type} adaptive curve: {fig_path}")
    
    # 绘制所有攻击类型的对比图
    plt.figure(figsize=(14, 10))

    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']

    for i, attack_type in enumerate(attack_types):
        if not all_results[attack_type]:
            continue

        # 根据攻击类型选择相应的攻击率数据
        if attack_type == "Modified Attributes":
            rates_to_plot = attack_rates_low
        else:
            rates_to_plot = attack_rates_high

        rates = sorted([rate for rate in rates_to_plot if rate in all_results[attack_type]])
        match_rates = [all_results[attack_type][rate]['match_rate'] for rate in rates]

        plt.plot(rates, match_rates,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2, markersize=6,
                label=attack_type)

    # 横坐标和纵坐标设置
    all_rates = []
    all_match_rates = []
    for attack_type in attack_types:
        if all_results[attack_type]:
            if attack_type == "Modified Attributes":
                rates_to_plot = attack_rates_low
            else:
                rates_to_plot = attack_rates_high

            rates_in_data = [rate for rate in rates_to_plot if rate in all_results[attack_type]]
            all_rates.extend(rates_in_data)
            all_match_rates.extend([all_results[attack_type][rate]['match_rate'] for rate in rates_in_data])

    if all_rates and all_match_rates:
        y_min, y_max = min(all_match_rates), max(all_match_rates)

        # 对比图横坐标设置为0到1的范围，显示所有数据点
        plt.xlim(0, 1)
        # 收集所有唯一的攻击率值并排序
        unique_rates = sorted(set(all_rates))
        plt.xticks(unique_rates)

        # 纵坐标自适应，添加边距
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        plt.ylim(0, 1.2)

    plt.xlabel('Attack Rate', fontsize=12)
    plt.ylabel('WER(%)', fontsize=12)
    plt.title('All Attack Types - Watermark Extraction Rate Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower left')
    plt.grid(True, alpha=0.3)

    # 保存对比图
    comparison_path = os.path.join(fig_dir, 'all_attacks_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison chart: {comparison_path}")
    
    # 保存汇总结果
    summary_path = os.path.join(result_dir, 'attack_test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nTesting completed!")
    print(f"Results saved to: {result_dir}")
    print(f"Images saved to: {fig_dir}")
    print(f"Summary results: {summary_path}")
    
    # 最终清理所有GPU资源
    try:
        print("Performing final GPU cleanup...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        print("Final GPU cleanup completed")
    except Exception as e:
        print(f"Final cleanup warning: {e}")
    
    return all_results

if __name__ == "__main__":
    # 运行RAG攻击效果测试
    test_attack_effects_RAG()
    