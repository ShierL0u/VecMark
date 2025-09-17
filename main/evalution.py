import pandas as pd
import numpy as np
import gc
import os
import re
import json
import torch
from tqdm import tqdm
from utils1 import to_original_path
from scipy.special import rel_entr
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import jensenshannon
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)     # 显示所有行
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.6f}'.format)

def load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"):
    """Load the base model and tokenizer only once"""
    try:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def create_answer_generate_prompt() -> PromptTemplate:
    combined_template = (
        """
        ROLE:
        You are a factual answer generator using RAG datasets.

        TASK:
        Output only the most similar original content from the provided context that matches the question.

        RULES:

        Output nothing but the most similar original text segment from the context.
        If no similar content exists in the context, output ONLY "No relevant information found".
        Never provide explanations, rephrasing, summaries, or interpretations.
        Never add any additional text beyond the extracted original content or the specified phrase when no relevant information is found.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        OUTPUT:
        Most similar original text segment from context OR "No relevant information found"
        """
    )
    return PromptTemplate(
        template=combined_template,
        input_variables=["context", "question"]  # 必须包含context变量
    )


def create_formatted_questions(answer):

    question = f"Output the contents of the RAG dataset that most closely resemble the following text : {answer}"
    return question

def extract_thinking_answer(text, model):
    if model == "qwen3:8b":
        pattern = r'</think>\n\n(.*)'
    elif model == "deepseek-r1:latest":
        pattern = r'</think>\n(.*)'
    else:
        return text
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 返回提取的内容并去除首尾空白
    return None  # 如果没有找到匹配的模式

def create_RS_judge_prompt(version=2) -> ChatPromptTemplate:
    judge_prompt_v1 = ("""You are an expert evaluator assessing the naturalness and reasonableness of watermarked text within relevant documents.

TASK: Rate how naturally and reasonably the watermark text fits within the context of the provided documents, with specific focus on its appropriateness within the document's contextual framework.

EVALUATION CRITERIA:

Content relevance: Does the watermark text logically connect with the document topics and context?
Linguistic naturalness: Does the text flow naturally with the surrounding content in context?
Contextual appropriateness: Is the watermark text appropriate and reasonable for the document's specific context and domain?

SCORING SCALE:

0.9-1.0: Perfectly natural, seamlessly integrated, and highly reasonable in context
0.7-0.8: Very natural, minor inconsistencies, mostly reasonable in context
0.5-0.6: Moderately natural, some noticeable issues, reasonably appropriate in context
0.3-0.4: Somewhat unnatural, clear inconsistencies, marginally reasonable in context
0.1-0.2: Very unnatural, major problems, barely reasonable in context
0.0: Completely unnatural, completely out of place, unreasonable in context

INSTRUCTIONS:

Carefully analyze the watermark text's reasonableness within the specific contextual framework of the provided documents
Consider all evaluation criteria with emphasis on contextual appropriateness
Provide a precise decimal score between 0.0 and 1.0 based on the level of reasonableness within context
Return ONLY the numerical score, no explanations or additional text

SCORE:""")

    judge_prompt_v2 = ("""You are an expert evaluator assessing the naturalness and reasonableness of watermarked text within relevant documents.

TASK: Rate how naturally and reasonably the watermark text fits within the context of the provided documents, with specific focus on its appropriateness within the document's contextual framework.

EVALUATION CRITERIA:

Content relevance: Does the watermark text logically connect with the document topics and context?
Linguistic naturalness: Does the text flow naturally with the surrounding content in context?
Contextual appropriateness: Is the watermark text appropriate and reasonable for the document's specific context and domain?

SCORING SCALE:

0.9-1.0: If and only if the WATERMARK TEXT differs from Document 0 in the RELEVANT DOCUMENTS FOR CONTEXT in only a few words. 
0.8-0.9: Extremely natural, fully aligned with context logic/style, seamlessly integrated, and highly reasonable.
0.6-0.8: Mostly natural, generally in line with context, with 1-2 minor inconsistencies; basically reasonable, no core logical conflicts.
0.5-0.6: Somewhat unnatural, with obvious expression flaws and 2-3 issues; barely meets basic rationality, with minor logical gaps.
0.2-0.4: Very unnatural, with chaotic expression and obvious inconsistencies; little rationality, only loosely related to the theme.
0.0-0.1: Completely unnatural, inconsistent with context in style/logic/theme; no rationality, disrupts context integrity.

INSTRUCTIONS:

Carefully analyze the watermark text's reasonableness within the specific contextual framework of the provided documents
Consider all evaluation criteria with emphasis on contextual appropriateness
Provide a precise decimal score between 0.0 and 1.0 based on the level of reasonableness within context
Return ONLY the numerical score, no explanations or additional text

SCORE:""")

    if version == 1:
        system_template = judge_prompt_v1
    elif version == 2:
        system_template = judge_prompt_v2
    else:
        raise ValueError(f"Invalid version: {version}")

    user_template = ('''WATERMARK TEXT TO EVALUATE:
{watermarked_text}

RELEVANT DOCUMENTS FOR CONTEXT (First confirm Document 1-Document 5; do not confuse their order):
{docs_text}''')

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def create_CDPA_judge_prompt():
    system_template = ("""Given two sentences, determine if they convey the same meaning. 
    If they are similar in meaning, return 'yes'; otherwise, return 'no'. 
    The following situations are also considered as the two sentences expressing the same meaning: 
    1. One sentence includes the meaning expressed in the other sentence. 
    2. The two sentences express the same central idea but in different ways. 

    Output: 'yes' or 'no' only, No explanations, no extra text.
    """)
    
    user_template = ("""
    Sentence 1: {clean_answer} 
    Sentence 2: {watermarked_answer}
    """)
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def calculate_kl_divergence(p, q, epsilon=1e-12):
    # Ensure non-zero probabilities for KL and JSD calculation by adding epsilon
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    
    # KL divergence (p || q)
    kl_divergence = np.sum(rel_entr(p, q))

    return kl_divergence

def calculate_distributions(column):
    # Calculate frequency distribution for a column
    value_counts = column.value_counts(normalize=True)
    return value_counts.index, value_counts.values

def compute_ppl(
    file_path: str,                
    text_column: str,
    model_name: str = "../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",             
    ctx_length: int = 2048,
    device: str = "cuda"
) -> float:

    data = pd.read_csv(file_path)

    # Load model and tokenizer
    model, tokenizer = load_model(model_name=model_name)
    
    try:
        text = " ".join(data[text_column].astype(str).tolist())

        encodings = tokenizer(text, return_tensors="pt").to(device)
        seq_len = encodings.input_ids.size(1)

        max_length = ctx_length
        stride = ctx_length
        nlls = []
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, seq_len, stride), desc="Eval ppl", miniters=max(1, (seq_len // stride) // 10)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                loss = model(input_ids, labels=target_ids).loss
            nlls.append(loss)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean()).item()
        
    finally:
        # Comprehensive GPU memory release
        print("Unloading model and freeing GPU memory...")
        
        # Move model to CPU first to free GPU memory
        if model is not None:
            try:
                # Move model to CPU before deletion
                model.cpu()
                # Clear model cache
                if hasattr(model, 'generation_config'):
                    delattr(model, 'generation_config')
            except:
                pass
            del model
            
        if tokenizer is not None:
            del tokenizer
        
        # Force garbage collection
        gc.collect()
        
        # Multiple rounds of GPU cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Clear again
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        # Final garbage collection
        gc.collect()
        
        print(f"Model unloading completed. GPU memory freed.")
        if torch.cuda.is_available():
            print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
    
    return ppl

def get_ollama_llm(model_name: str = "llama3:8b", **kwargs):
    """创建Ollama LLM实例"""
    defaults = {
        "temperature": 0.3,
        "top_p": 0.7,
        "timeout": 120
    }
    defaults.update(kwargs)
    return ChatOllama(model=model_name, **defaults)

def extract_dataset_name(csv_path: str) -> str:
    """根据csv_path自动生成dataset_name"""
    # 获取文件名(不包含路径)
    filename = os.path.basename(csv_path)
    
    # 移除.csv扩展名
    dataset_name = os.path.splitext(filename)[0]
    
    return dataset_name

def create_RAGdb(file_path, dataset_name, column_name: str, emb_model="bge-m3:latest", db_name="RAG_db"):

    vector_db_dir = f"vector_database/{dataset_name}"
    embedding_model = OllamaEmbeddings(model=emb_model)  # 提前创建嵌入模型

    if os.path.exists(vector_db_dir) and os.path.isdir(vector_db_dir):
        vector_db = Chroma(
            persist_directory=vector_db_dir,
            collection_name=db_name,
            embedding_function=embedding_model  # 加载时指定嵌入函数
        )
        print(f"Success load vector_db {dataset_name}")
        return vector_db

    # 如果不存在数据库或加载失败，则重新创建
    print(f"create vector_db {dataset_name}")
    csv_loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ','})
    documents = csv_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    texts = text_splitter.split_documents(documents)
    
    # 创建数据库时指定嵌入函数
    vector_db = Chroma.from_documents(
        texts,
        embedding_model,  # 创建时指定嵌入函数
        collection_name=db_name,
        persist_directory=vector_db_dir
    )
    return vector_db

def create_rag_chain(vector_db, rag_model: str = "qwen:14b", source_documents=False):
    """创建RAG问答链"""
    # 创建简单的问答提示模板
    answer_prompt = create_answer_generate_prompt()
    
    # 创建RAG链
    rag_chain = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(model_name=rag_model),
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),  # 获取top5相关文档
        chain_type_kwargs={"prompt": answer_prompt},
        return_source_documents=source_documents
    )
    
    return rag_chain

def judge_relevance(watermarked_text: str, retrieved_contexts: list, judge_model_name: str, judge_llm) -> float:
    """使用评估模型对水印文本在检索文档中的合理性进行评分"""
    
    # 构建评估提示，将5个文档分别列出
    docs_sections = []
    for i, doc in enumerate(retrieved_contexts, 1):
        docs_sections.append(f"Document {i}:\n{doc}")
    
    docs_text = "\n\n".join(docs_sections)
    
    response = judge_llm.invoke({"watermarked_text": watermarked_text, "docs_text": docs_text})
    
    # 提取评分
    score_text = extract_thinking_answer(response.content, judge_model_name)
    try:
        match = re.search(r'-?\d+\.?\d*', score_text)
        score_text = match.group()  # 获取匹配到的数字字符串
    except Exception as e:
        print(f"Error in judge_relevance: {e}")
        # 若未提取到数字，可设置默认值或抛出异常
        score_text = "0.0"
    # 确保评分在0-1范围内
    score = max(0.0, min(1.0, float(score_text)))
    return score

def extract_watermark_data(csv_path: str, method_name: str):
    """根据方法名提取水印嵌入信息"""
    dataset_type = csv_path.split('_')[2]
    watermarked_data = []
    if method_name == "VectorMark":
        # VectorMark: 从映射文件中提取水印信息
        mapping_file = csv_path.replace('.csv', '_rewrite_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            for key, value in mapping_data.items():
                if dataset_type == "nfcorpus":
                    watermarked_data.append(value['text']['overall_rewritten_sentence'])
                elif dataset_type == "winemag50k":
                    watermarked_data.append(value['description']['overall_rewritten_sentence'])
            
        else:
            print(f"Warning: Mapping file {mapping_file} not found")
    
    elif method_name == "RAGWM":
        # RAGWM: 从CSV中提取watermark_unit == True的行的text值
        data = pd.read_csv(csv_path)
        dataset_type = dataset_type[:-4]
        for idx, row in data.iterrows():
            # 返回watermark_embedded为True的行的text值
            if dataset_type == "nfcorpus" and row['watermark_embedded'] == True:
                watermarked_data.append(row['text'])
            elif dataset_type == "winemag50k" and row['watermark_embedded'] == True:
                watermarked_data.append(row['description'])
    
    elif method_name == "WARD":
        # WARD: 从stats文件中提取rewritten_text
        stats_file = csv_path.replace('.csv', '_rewrite_mapping.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
            
            # 提取被水印的文本内容
            for key, value in stats_data.items():
                if dataset_type == "nfcorpus":  
                    watermarked_data.append(value['text']['rewritten_text'])
                elif dataset_type == "winemag50k":
                    watermarked_data.append(value['description']['rewritten_text'])
            
        else:
            print(f"Warning: Stats file {stats_file} not found")
    
    else:
        print(f"Warning: Unknown method {method_name}")
    
    return watermarked_data

def extract_clean_data(all_df: pd.DataFrame, watermarked_data: list, column_name: str, max_num: int = 100):
    """提取max_num条clean data, 即在all_df中而不在watermarked_data中的data"""
    clean_data = []
    tmp_num = 0
    for idx, row in all_df.iterrows():
        if row[column_name] in watermarked_data:
            continue
        clean_data.append(row[column_name])
        tmp_num += 1
        if tmp_num >= max_num:
            break
    return clean_data

def compute_RS(csv_path: str, method_name, judge_llm, judge_model_name: str = "qwen3:8b",
               model_RAG: str = "qwen:14b", 
               text_column: str = "text", emb_model: str = "bge-m3:latest", 
               rag_dataset: str = "clean", 
               num_samples: int = -1) -> float:
    """
    计算RAG系统的合理性评分 (Relevance Score)
    
    Args:
        csv_path: CSV数据集路径
        method_name: 水印方法名称
        judge_llm: 评估模型实例
        model_RAG: RAG模型名称
        text_column: 文本列名
        emb_model: 嵌入模型名称
    
    Returns:
        float: 平均合理性评分 (0-1)
    """
    
    print(f"\n Computing RS for {csv_path}...")
    print(f"RAG Model: {model_RAG}")
    print(f"Judge Model: {judge_model_name}")
    
    # 自动生成dataset_name
    dataset_name = extract_dataset_name(csv_path)
    print(f"Dataset name: {dataset_name}")

    if rag_dataset == "clean":
        original_csv_path, original_dataset_name = to_original_path(csv_path)
        data = pd.read_csv(original_csv_path)
        print(f"Loaded {len(data)} from {original_csv_path}")
    # 读取数据
    elif rag_dataset == "watermarked":
        data = pd.read_csv(csv_path)
        print(f"Loaded {len(data)} from {csv_path}")
    
    # 提取水印数据
    watermarked_texts = extract_watermark_data(csv_path, method_name)

    # 小数据测试
    if num_samples > 0:
        watermarked_texts = watermarked_texts[:num_samples]

    print(f"Found {len(watermarked_texts)} watermarked samples using {method_name} method")
    
    if not watermarked_texts:
        print("No watermarked data found, returning default score 0.5")
        return 0.5
    
    print(f"Processing {len(watermarked_texts)} watermarked texts")
    
    # 创建RAG数据库
    print("Creating RAG database...")
    vector_db = create_RAGdb(file_path=csv_path, dataset_name=dataset_name, 
                            column_name=text_column, emb_model=emb_model)
    
    # 创建RAG链
    print("Creating RAG chain...")
    rag_chain = create_rag_chain(vector_db, model_RAG, source_documents=True)
    
    # 评估每个水印文本
    scores = []
    print(f"Evaluating {len(watermarked_texts)} watermarked samples...")
    
    for watermarked_text in tqdm(watermarked_texts, desc=f"RS {dataset_name}"):

        watermarked_text = str(watermarked_text)
        
        # 使用RAG系统检索相关文档
        result = rag_chain.invoke({"query": watermarked_text})
        retrieved_docs = result['source_documents']
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        # 从向量数据库查询
        # retrieved_docs = vector_db.similarity_search(watermarked_text, k=5)
        
        # 使用评估模型评分
        score = judge_relevance(watermarked_text,
                                retrieved_contexts, 
                                judge_model_name,
                                judge_llm)
        scores.append(score)
            
    
    # 计算平均评分
    avg_score = np.mean(scores) if scores else 0.0
    
    print(f"RS Evaluation completed:")
    print(f"  - Total watermarked samples: {len(scores)}")
    print(f"  - Average RS score: {avg_score:.4f}")
    print(f"  - Score range: {min(scores):.3f} - {max(scores):.3f}")
    
    return avg_score

def evalution_onRAG(file_path, text_column, method_name="",
                    judge_llm=None, 
                    ppl_model_name:str="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", 
                    judge_model_name:str="qwen3:8b",
                    num_samples=-1):

    # ppl = compute_ppl(
    #     file_path,        
    #     text_column=text_column,
    #     model_name=ppl_model_name,       
    #     ctx_length=2048
    # )

    ppl = 0

    RS = compute_RS(file_path, 
                    method_name=method_name, 
                    judge_llm=judge_llm, 
                    text_column=text_column, 
                    judge_model_name=judge_model_name,
                    num_samples=num_samples)

    # return ppl, RS
    return ppl, RS

def evalution_onDataset(df_original, df_transformed):
    '''
        评估数据集的 KL散度 和 average deviation
    '''
   
    numeric_columns = df_original.select_dtypes(include=['int', 'float']).columns
    category_columns = df_original.select_dtypes(include=['object']).columns
    
    # Store results
    # results = {'Mean Difference': [], 'Variance Difference': [], 'KL Divergence': [], 'JSD Divergence': []}
    results = {
        'KL Divergence': [],
        'Average Deviation': []
    }

    # 1. Calculate mean, variance, KL, and JSD for numeric columns
    # for col in tqdm(numeric_columns, desc="Processing numeric cols"):
    for col in numeric_columns:
        # Mean and standard deviation
        original_mean = df_original[col].mean()
        original_std = df_original[col].std()

        transformed_mean = df_transformed[col].mean()
        transformed_std = df_transformed[col].std()

        # Absolute differences
        mean_diff_abs = abs(transformed_mean - original_mean)
        std_diff_abs = abs(transformed_std - original_std)

        # Calculate distributions (for KL and JSD)
        original_values, original_dist = calculate_distributions(df_original[col])
        transformed_values, transformed_dist = calculate_distributions(df_transformed[col])
        
        # Align distributions for KL and JSD (only keep transformed values that exist in original)
        # transformed_dist = [transformed_dist[list(transformed_values).index(v)] if v in transformed_values else 0 for v in original_values]
        value_to_transformed = dict(zip(transformed_values, transformed_dist))
        transformed_dist = [value_to_transformed.get(v, 0) for v in original_values]

        # Normalize distributions (to avoid zero division)
        original_dist = np.array(original_dist) / np.sum(original_dist)
        transformed_dist = np.array(transformed_dist) / np.sum(transformed_dist)
        
        # Calculate KL divergences
        kl_div = calculate_kl_divergence(original_dist, transformed_dist)  

        avg_dev = mean_diff_abs / abs(original_mean) if original_mean != 0 else np.nan      

        results['KL Divergence'].append(kl_div)
        results['Average Deviation'].append(avg_dev)
        
    # 2. Calculate KL and JSD for category columns
    # for col in tqdm(category_columns, desc="Processing category cols"):
    for col in category_columns:
        
        original_values, original_dist = calculate_distributions(df_original[col])
        transformed_values, transformed_dist = calculate_distributions(df_transformed[col])

        # Restrict transformed distribution to original values
        # transformed_dist = [transformed_dist[list(transformed_values).index(v)] if v in transformed_values else 0 for v in original_values]
        value_to_transformed = dict(zip(transformed_values, transformed_dist))
        transformed_dist = [value_to_transformed.get(v, 0) for v in original_values]

        # Normalize distributions
        original_dist = np.array(original_dist) / np.sum(original_dist)
        transformed_dist = np.array(transformed_dist) / np.sum(transformed_dist)
        
        # Calculate KL and JSD divergences
        kl_div = calculate_kl_divergence(original_dist, transformed_dist)

        results['KL Divergence'].append(kl_div)
        results['Average Deviation'].append(np.nan)
    # Create final DataFrame for results
    index_with_types = [f"{col} ({df_original[col].dtype})" for col in numeric_columns] + \
                   [f"{col} ({df_original[col].dtype})" for col in category_columns]

    result_df = pd.DataFrame(results, index=index_with_types)
    mean_row = result_df.mean()
    mean_row.name = 'Average'
    result_df = pd.concat([result_df, mean_row.to_frame().T])

    return result_df

def test_onDataset():
    data_winemag50k = pd.read_csv('../dataset/winemag_sub_dataset_50k.csv')
    data_FCT100k = pd.read_csv('../dataset/FCT_100k.csv')
    data_nfcorpus = pd.read_csv('../dataset/nfcorpus_corpus.csv')
    winemag50k_attributes = data_winemag50k.columns[1:].tolist() 
    FCT100k_attributes = data_FCT100k.columns[1:].tolist() 
    nfcorpus_attributes = data_nfcorpus.columns[1:].tolist() # 不考虑id列
    
    VectorMark_winemag50k = pd.read_csv('../dataset/watermark_dataset/VectorMark_winemag50k_g587.csv')
    
    result = evalution_onDataset(data_winemag50k[winemag50k_attributes], VectorMark_winemag50k[winemag50k_attributes])
    print("------------VectorMark winemag50k------------\n", result)
    
    VectorMark_FCT100k = pd.read_csv('../dataset/watermark_dataset/VectorMark_FCT100k_g1721.csv')
    result = evalution_onDataset(data_FCT100k[FCT100k_attributes], VectorMark_FCT100k[FCT100k_attributes])
    print("--------------VectorMark FCT100k-------------\n", result)
   
    VectorMark_nfcorpus = pd.read_csv('../dataset/watermark_dataset/VectorMark_nfcorpus_g37.csv')
    result = evalution_onDataset(data_nfcorpus[nfcorpus_attributes], VectorMark_nfcorpus[nfcorpus_attributes])
    print("-------------VectorMark nfcorpus-------------\n", result)

    RRWC_FCT100k = pd.read_csv('../dataset/watermark_dataset/RRWC_FCT100k.csv')
    result = evalution_onDataset(data_FCT100k[FCT100k_attributes], RRWC_FCT100k[FCT100k_attributes])
    print("-------------RRWC FCT100k-------------\n", result)

    RRWC_winemag50k = pd.read_csv('../dataset/watermark_dataset/RRWC_winemag50k.csv')
    result = evalution_onDataset(data_winemag50k[winemag50k_attributes], RRWC_winemag50k[winemag50k_attributes])
    print("-------------RRWC winemag50k-------------\n", result)

def generate_rag_results(method_name: str, dataset_name: str, 
                        clean_rag_chain, watermarked_rag_chain,
                        clean_vector_db, watermarked_vector_db, 
                        clean_texts: list,
                        clean_RAG_model: str = "qwen:14b",
                        watermarked_RAG_model: str = "deepseek-r1:latest",
                        output_dir: str = "rag_results", force_regenerate: bool = False) -> tuple:
    """
    生成Clean RAG和Watermarked RAG的结果，并保存到JSON文件中
    如果结果文件已存在且force_regenerate为False，则直接加载已有结果

    Args:
        method_name: 水印方法名称
        dataset_name: 数据集名称
        clean_rag_chain: Clean RAG链
        watermarked_rag_chain: Watermarked RAG链
        clean_vector_db: Clean向量数据库
        watermarked_vector_db: Watermarked向量数据库
        clean_texts: 部分干净文本列表
        output_dir: 输出目录
        force_regenerate: 是否强制重新生成结果, 默认为False

    Returns:
        tuple: (clean_results, watermarked_results)
    """
    import os
    import json

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义文件路径
    clean_json_path = os.path.join(output_dir, f"{method_name}_{dataset_name}_CleanResult.json")
    watermarked_json_path = os.path.join(output_dir, f"{method_name}_{dataset_name}_WatermarkedResult.json")

    # 检查结果文件是否已存在
    all_files_exist = all([
        os.path.exists(clean_json_path),
        os.path.exists(watermarked_json_path)
    ])

    if all_files_exist and not force_regenerate:

        try:
            # 加载Clean结果
            with open(clean_json_path, 'r', encoding='utf-8') as f:
                clean_results = json.load(f)

            # 加载Watermarked结果
            with open(watermarked_json_path, 'r', encoding='utf-8') as f:
                watermarked_results = json.load(f)

            print(f"loaded {len(clean_results)} clean results and {len(watermarked_results)} watermarked results")
            return clean_results, watermarked_results

        except Exception as e:
            print(f"load results file error: {e}")
    else:
        if all_files_exist:
            print("force to generate new results...")
        else:
            print("results file not exist, start to generate new results...")

    # 如果需要生成新结果
    clean_results = []
    watermarked_results = []

    print(f"Generating RAG results for {len(clean_texts)} queries...")

    for i, query_text in enumerate(tqdm(clean_texts, desc="Generating RAG results")):
        query_text = str(query_text)
        query = f"QUESTION: {create_formatted_questions(query_text)}"
        # query = query_text

        # Clean RAG结果
        clean_result = clean_rag_chain.invoke({"query": query})

        # 获取Clean RAG检索到的文档
        clean_answer = extract_thinking_answer(clean_result['result'], clean_RAG_model)
        clean_retrieved_docs = clean_result['source_documents']
        # clean_retrieved_docs = clean_vector_db.similarity_search(query_text, k=5)
        clean_retrieval_texts = [doc.metadata['row'] for doc in clean_retrieved_docs]

        # Watermarked RAG结果
        watermarked_result = watermarked_rag_chain.invoke({"query": query})

        # 获取Watermarked RAG检索到的文档
        watermarked_answer = extract_thinking_answer(watermarked_result['result'], watermarked_RAG_model)
        watermarked_retrieved_docs = watermarked_result['source_documents']
        # watermarked_retrieved_docs = watermarked_vector_db.similarity_search(query_text, k=5)
        watermarked_retrieval_texts = [doc.metadata['row'] for doc in watermarked_retrieved_docs]


        # 存储Clean结果
        clean_result_data = {
            "query_index": i,
            "query_text": query_text,
            "answer": clean_answer,
            "retrieved_texts": clean_retrieval_texts
        }
        clean_results.append(clean_result_data)

        # 存储Watermarked结果
        watermarked_result_data = {
            "query_index": i,
            "query_text": query_text,
            "answer": watermarked_answer,
            "retrieved_texts": watermarked_retrieval_texts
        }
        watermarked_results.append(watermarked_result_data)

    # 保存Clean结果到JSON文件
    with open(clean_json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    print(f"Clean RAG results saved to: {clean_json_path}")

    # 保存Watermarked结果到JSON文件
    with open(watermarked_json_path, 'w', encoding='utf-8') as f:
        json.dump(watermarked_results, f, ensure_ascii=False, indent=2)
    print(f"Watermarked RAG results saved to: {watermarked_json_path}")

    print(f"saved {len(clean_results)} clean results and {len(watermarked_results)} watermarked results")
    return clean_results, watermarked_results

def evaluate_rag_alignment(csv_path: str, method_name: str = "",
                          clean_RAG_model: str = "qwen:14b",
                          watermarked_RAG_model: str = "qwen:14b",
                          emb_model: str = "bge-m3:latest", text_column: str = "text",
                          CDPA_judge_chain = None, CDPA_judge_model: str = "qwen3:8b", 
                          num_samples: int = -1, force_regenerate: bool = False) -> tuple:
    """
    评估RAG系统的对齐性指标(CDPA和CIRA)
    自动生成Clean RAG和Watermarked RAG的结果进行比较

    Args:
        csv_path: 水印数据集CSV文件路径
        method_name: 水印方法名称
        model_RAG: RAG模型名称
        emb_model: 嵌入模型名称
        text_column: 文本列名
        CDPA_judge_chain: CDPA判断模型名称
        num_samples: 评估样本数量
        force_regenerate: 是否强制重新生成RAG结果，默认为False

    Returns:
        dict: 包含CDPA和CIRA分数的字典
    """
    print(f"\n evaluate rag alignment - method: {method_name}")
    print(f"watermarked dataset: {csv_path}")
    
    # 自动生成dataset_name
    dataset_name = extract_dataset_name(csv_path)
    print(f"Dataset name: {dataset_name}")
    
    # 提取水印数据
    watermarked_texts = extract_watermark_data(csv_path, method_name)
        
    print(f"Processing {len(watermarked_texts)} watermarked samples")
    
    # 创建Clean RAG数据库(使用原始数据集)
    print("Creating Clean RAG database...")
    original_csv_path, original_dataset_name = to_original_path(csv_path)
    if not original_csv_path:
        print("Warning: Cannot find original dataset path")
        return {"CDPA": 0.0, "CIRA": 0.0, "error": "Original dataset not found"}
        
    original_df = pd.read_csv(original_csv_path)
    clean_texts = extract_clean_data(all_df=original_df,
                                     watermarked_data=watermarked_texts,
                                     column_name=text_column,
                                     max_num=num_samples)
    if not clean_texts:
        print("No clean data found, returning default scores")
        return {"CDPA": 0.0, "CIRA": 0.0, "error": "No clean data found"}

    clean_vector_db = create_RAGdb(file_path=original_csv_path,
                                    dataset_name=f"{original_dataset_name}_clean", 
                                    emb_model=emb_model, 
                                    column_name=text_column)
    clean_rag_chain = create_rag_chain(clean_vector_db, clean_RAG_model, source_documents=True)
    # clean_rag_chain = create_rag_chain(clean_vector_db, model_RAG)
    
    # 创建Watermarked RAG数据库(使用水印数据集)
    print("Creating Watermarked RAG database...")
    watermarked_vector_db = create_RAGdb(file_path=csv_path,
                                            dataset_name=dataset_name,
                                            emb_model=emb_model,
                                            column_name=text_column)
    watermarked_rag_chain = create_rag_chain(watermarked_vector_db, watermarked_RAG_model, source_documents=True)
    
    # 使用新函数生成RAG结果并保存到JSON文件
    clean_results, watermarked_results = generate_rag_results(
        method_name=method_name,
        dataset_name=dataset_name,
        clean_rag_chain=clean_rag_chain,
        watermarked_rag_chain=watermarked_rag_chain,
        clean_vector_db=clean_vector_db,
        watermarked_vector_db=watermarked_vector_db,
        clean_RAG_model=clean_RAG_model,
        watermarked_RAG_model=watermarked_RAG_model,
        clean_texts=clean_texts,
        output_dir="data/rag_result",
        force_regenerate=force_regenerate
    )

    # 存储每个查询的CDPA和CIRA分数
    individual_cdpa_scores = []
    individual_cira_scores = []

    # 计算CDPA和CIRA分数
    for i, (clean_result, watermarked_result) in enumerate(zip(clean_results, watermarked_results)):
        clean_answer = clean_result["answer"]
        clean_retrieval_texts = clean_result["retrieved_texts"]

        watermarked_answer = watermarked_result["answer"]
        watermarked_retrieval_texts = watermarked_result["retrieved_texts"]

        # 为当前查询计算CDPA
        query_cdpa = CDPA_judge_chain.invoke({"clean_answer": clean_answer,
                                                "watermarked_answer": watermarked_answer})
        cdpa_answer = extract_thinking_answer(query_cdpa.content, CDPA_judge_model)
        if cdpa_answer == "yes":
            cdpa_value = 1.0
        else:
            cdpa_value = 0.0
        individual_cdpa_scores.append(cdpa_value)

        # 为当前查询计算CIRA
        if clean_retrieval_texts and watermarked_retrieval_texts:
            same_retrievals = 0
            for clean_doc, watermarked_doc in zip(clean_retrieval_texts, watermarked_retrieval_texts):
                if clean_doc == watermarked_doc:
                    same_retrievals += 1
        else:
            same_retrievals = 0
            query_cira = 0.0

        individual_cira_scores.append(same_retrievals)

    CDPA_count = sum(individual_cdpa_scores)
    CIRA_count = sum(individual_cira_scores)
    CDPA_rate = CDPA_count/len(individual_cdpa_scores)
    CIRA_rate = CIRA_count/(len(individual_cira_scores)*len(clean_retrieval_texts))
    # 打印结果统计
    print(f"\nRAG Alignment Evaluation Results:")
    print(f"  - Method: {method_name}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Total queries: {len(watermarked_texts)}")
    print(f"  - CDPA count: {int(CDPA_count)}")
    print(f"  - CDPA rate: {CDPA_rate:.4%}")
    print(f"  - CIRA count: {int(CIRA_count)}")
    print(f"  - CIRA rate: {CIRA_rate:.4%}")
    
    return CDPA_rate, CIRA_rate


def test_onRAG_PPL_RS(num_samples=-1, version=2, judge_model="qwen3:8b"):
    """Test RAG perplexity evaluation with table output"""
    # 存储结果的列表
    results_list = []

    print("Evaluating perplexity results...")
    
    # 创建judge_llm实例，整个评估过程只创建一次
    print("Creating judge model instance...")
    judge_llm = get_ollama_llm(model_name=judge_model, temperature=0.1)
    judge_llm_chain = create_RS_judge_prompt(version=version) | judge_llm
    print("Judge model created successfully")
    
    # 评估参数列表：每个参数组用大括号包裹，包含(方法名, 数据集路径, 数据集名称, 文本列名)
    eval_params = [
        {
            'method': 'VectorMark', 
            'file_path': '../dataset/watermark_dataset/VectorMark_nfcorpus_g37.csv', 
            'dataset': 'NFCorpus', 'text_col': 'text'
        },
        {
            'method': 'VectorMark', 
            'file_path': '../dataset/watermark_dataset/VectorMark_winemag50k_g587.csv', 
            'dataset': 'Winemag50k', 'text_col': 'description'
        },
        {
            'method': 'RAGWM', 
            'file_path': '../dataset/watermark_dataset/RAGWM_nfcorpus.csv', 
            'dataset': 'NFCorpus', 'text_col': 'text'
        },
        {
            'method': 'RAGWM', 
            'file_path': '../dataset/watermark_dataset/RAGWM_winemag50k.csv', 
            'dataset': 'Winemag50k', 
            'text_col': 'description'
        },
        {
            'method': 'WARD', 
            'file_path': '../dataset/watermark_dataset/WARD_nfcorpus_191.csv', 
            'dataset': 'NFCorpus', 
            'text_col': 'text'
        },
        {
            'method': 'WARD', 
            'file_path': '../dataset/watermark_dataset/WARD_winemag50k_127.csv', 
            'dataset': 'Winemag50k', 
            'text_col': 'description'
        }
    ]
    
    # 使用tqdm遍历参数列表进行评估
    for params in tqdm(eval_params, desc="Evaluating RAG models"):
        ppl, rs = evalution_onRAG(
            file_path=params['file_path'],
            text_column=params['text_col'],
            method_name=params['method'],
            judge_llm=judge_llm_chain,
            judge_model_name=judge_model,
            num_samples=num_samples
        )
        results_list.append({
            'Method': params['method'],
            'Dataset': params['dataset'],
            'Text_Column': params['text_col'],
            'Perplexity': ppl,
            'Rationality Score': rs
        })
    
    # 创建结果表格
    results_table = pd.DataFrame(results_list)
    
    # 格式化输出表格
    print("\n" + "="*80)
    print("RAG Evaluation Results")
    print("="*80)
    print(results_table.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return results_table
    

def test_onRAG_CDPA_CIRA(CDPA_judge_model:str="qwen3:8b", num_samples:int=-1, force_regenerate:bool=False):
    """Test RAG perplexity evaluation with table output and tqdm progress bar"""
    # Store results in list for table creation
    results_list = []

    print("Evaluating perplexity results...")
    
    # 创建judge_llm实例，整个评估过程只创建一次
    print("Creating judge model instance...")
    CDPA_judge = get_ollama_llm(model_name=CDPA_judge_model, temperature=0.1)
    CDPA_judge_model_chain = create_CDPA_judge_prompt() | CDPA_judge
    print("Judge model created successfully")
    
    # 定义所有评估任务参数
    evaluation_tasks = [
        {
            'method': 'VectorMark',
            'dataset': 'nfcorpus',
            'path': '../dataset/watermark_dataset/VectorMark_nfcorpus_g37.csv',
            'text_column': 'text'
        },
        {
            'method': 'VectorMark',
            'dataset': 'Winemag50k',
            'path': '../dataset/watermark_dataset/VectorMark_winemag50k_g587.csv',
            'text_column': 'description'
        },
        {
            'method': 'RAGWM',
            'dataset': 'nfcorpus',
            'path': '../dataset/watermark_dataset/RAGWM_nfcorpus.csv',
            'text_column': 'text'
        },
        {
            'method': 'RAGWM',
            'dataset': 'Winemag50k',
            'path': '../dataset/watermark_dataset/RAGWM_winemag50k.csv',
            'text_column': 'description'
        },
        {
            'method': 'WARD',
            'dataset': 'NFCorpus',
            'path': '../dataset/watermark_dataset/WARD_nfcorpus_191.csv',
            'text_column': 'text'
        },
        {
            'method': 'WARD',
            'dataset': 'winemag50k',
            'path': '../dataset/watermark_dataset/WARD_winemag50k_127.csv',
            'text_column': 'description'
        }
    ]
    
    # 使用tqdm创建进度条，遍历所有评估任务
    for task in tqdm(evaluation_tasks, desc="Evaluating datasets"):
        cdpa_score, cira_score = evaluate_rag_alignment(
            csv_path=task['path'],
            method_name=task['method'],
            CDPA_judge_chain=CDPA_judge_model_chain,
            CDPA_judge_model=CDPA_judge_model,
            text_column=task['text_column'],
            emb_model='bge-m3:latest',
            num_samples=num_samples,
            force_regenerate=force_regenerate
        )
        
        results_list.append({
            'Method': task['method'],
            'Dataset': task['dataset'],
            'Text_Column': task['text_column'],
            'CDPA': cdpa_score,
            'CIRA': cira_score
        })

    # Create results table
    results_table = pd.DataFrame(results_list)
    
    # Display table with formatted output
    print("\n" + "="*80)
    print("RAG Evaluation Results")
    print("="*80)
    print(results_table.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return results_table

if __name__ == "__main__":
    # 运行RAG对齐性评估测试
    # test_onRAG_CDPA_CIRA(num_samples=50, force_regenerate=False)
    
    # 运行原有的RAG评估
    test_onRAG_PPL_RS(num_samples=-1, version=2, judge_model="qwen3:8b")
    # test_onDataset()