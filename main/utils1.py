import hashlib
import random
from typing import List, Dict, Tuple, Union, Any
import json
import pandas as pd
import numpy as np
import torch
import logging
import re
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from KGW.KGW_claude import KGWWatermarkedLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA  
from langchain_community.document_loaders import DataFrameLoader, CSVLoader

def create_json_schema(name):
    question_json_schema = None
    if name == "wikipedia" or name == "nfcorpus":
        question_json_schema = {
            "title": "keywords_extract",
            "description": "keywords_extract",
            "type": "object",
            "properties": {
                "text": {
                    "type": "array",
                    "description": "the key word",
                },
            },
            "required": ["text"],
        }

    return question_json_schema

def select_OBJECT_fields(dataset_name):
    if dataset_name == "wikipedia":
        OBJECT_fields = ["text"]
    elif dataset_name == "nfcorpus":
        OBJECT_fields = ["text"]
    elif "winemag" in dataset_name:
        OBJECT_fields = ["description", "title"]
    elif dataset_name == "FCT" or dataset_name == "FCT100k":
        OBJECT_fields = []

    return OBJECT_fields

def remove_think_tags(text):
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL)

def hash_sig(sig: str, L: int = 16) -> List[int]:
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer")
    hash_hex = hashlib.sha256(sig.encode()).hexdigest()
    hash_bin = bin(int(hash_hex, 16))[2:].zfill(L)
    if L <= 256:
        return list(map(int, hash_bin[:L]))
    else:
        extended_bin = (hash_bin * ((L // 256) + 1))[:L]
        return list(map(int, extended_bin))
    
def is_same(hash_key_0, hash_key_1):
    if hash_key_0==hash_key_1:
        return "is same"
    return ""

def hash_int_test():
    K2="Keyword2"
    for bit_position in range(1,16):
        hash_key_0 = hash_int(K2, bit_position, 0)
        hash_key_1 = hash_int(K2, bit_position, 1)
        print(f"BP:{bit_position} | hash_key_0 {hash_key_0} | hash_key_1 {hash_key_1} {is_same(hash_key_0, hash_key_1)}")

def hash_int(key: str, BP: int, wm_bit: int, mod_num:int) -> int:
    '''需提前确定mod_num在该配置下是否存在冲突, 可用hash_int_test()进行测试'''

    concat = f"{key}{BP}{wm_bit}"
    hash_val = hashlib.sha256(concat.encode()).hexdigest()
    return int(hash_val, 16) % mod_num

def hash_value_key(value: str, key: str, mod: int) ->int:
    concat = f"{value}{key}"
    hash_val = hashlib.sha256(concat.encode()).hexdigest()
    return int(hash_val, 16) % mod

def hash_val(concat):
    hash_val = hashlib.sha256(concat.encode()).hexdigest()
    return hash_val

def hash_modlist(value: str, key: str, mod_list: List[int]) ->int:
    concat = f"{value}{key}"
    hash_val = hashlib.sha256(concat.encode()).hexdigest()
    for mod in mod_list:
        if int(hash_val, 16) % mod == 0:
            return 0
    return 1

def select_RAGtext_field(name):
    if name=="nfcorpus":
        output = ["text"]
    elif name=="winemag50k" or name=="winemag5k":
        output = ["text"]
        
    return output

def select_ka_ca(df: pd.DataFrame, dataset_name: str) -> Tuple[List[str], List[str], List[Tuple]]:
    predefined_columns = {
        "FCT": {
            "KA": ["id"],
            "CA": ["Aspect", "Slope", "Hillshade_9am", "Hillshade_Noon"],
            "A": [col for col in df.columns if not any(keyword in col for keyword in ["Soil", "Wilderness", "Cover", "id"])]
        },
        "FCT100k": {
            "KA": ["id"],
            "CA": ["Aspect", "Slope", "Hillshade_9am", "Hillshade_Noon"],
            "A": [col for col in df.columns if not any(keyword in col for keyword in ["Soil", "Wilderness", "Cover", "id"])]
        },
        "winemag5k": {
            "KA": ["id", "winery"],
            "CA": ["country", "price", "taster_name", "taster_twitter_handle"],
            "A": ["description", "points", "region_2", "title", "country", "price", "taster_name", "taster_twitter_handle"]
        },
        "winemag50k": {
            "KA": ["id", "winery"],
            "CA": ["country", "price", "taster_name", "taster_twitter_handle"],
            "A": ["description", "points", "region_2", "title", "country", "price", "taster_name", "taster_twitter_handle"]
        },
        "nfcorpus": {
            "KA": ["id"],
            "CA": ["title"],
            "A": ["text"]
        },
        "wikipedia": {
            "KA": ["id", "wiki_id"],
            "CA": ["title", "url", "views", "paragraph_id", "langs"],
            "A": ["text", "title", "url", "views", "paragraph_id", "langs"]

        }
        # 添加更多数据集配置...
    }
    
    config = predefined_columns.get(dataset_name, {})
    
    KA = config.get("KA", ["id"])  # 默认使用"id"作为关键属性
    CA = config.get("CA", [])
    A = config.get("A", [])
    
    KA = [col for col in KA if col in df.columns]
    CA = [col for col in CA if col in df.columns]
    A = [col for col in A if col in df.columns]
    
    # if not CA:
    #     CA_candidates = [col for col in df.columns if col not in KA]
    #     if dataset_name=="nfcorpus":
    #         CA = ["title"]
    #     else:
    #         CA = CA_candidates[:4]  # 取前4个作为默认
    if dataset_name=="nfcorpus":
        one_hot_masks = ["1"]
    else:
        one_hot_masks = ["1100", "1010", "1011", "0101"]
    one_hot_combinations = []
    for mask in one_hot_masks:
        combo = tuple(CA[i] if i < len(CA) and mask[i] == '1' else '' for i in range(len(one_hot_masks)))
        one_hot_combinations.append(combo)

    return KA, A, one_hot_combinations

# def sample_reference_context(df: pd.DataFrame, sample_ratio: float = 0.01, max_samples: int = 100, random_state: int = 42) -> list[dict]:
#     n_total = len(df)
#     n_sample = min(int(n_total * sample_ratio), max_samples)
#     sample_df = df.sample(n=n_sample, random_state=random_state)
#     context_entities = sample_df.to_dict(orient='records')
#     return context_entities


def compare_memory_usage(
    df: pd.DataFrame,
    savings_threshold: float = 0.5,
    cardinality_threshold: float = 0.5,
    verbose: bool = True
):
    detailed = {}
    suggested = []

    for col in df.select_dtypes(include=['object']).columns:
        orig_mem = df[col].memory_usage(deep=True)
        cat_col = df[col].astype('category')
        cat_mem = cat_col.memory_usage(deep=True)
        savings = (orig_mem - cat_mem) / orig_mem if orig_mem > 0 else 0
        nunique = df[col].nunique(dropna=True)
        n_rows = len(df[col].dropna())
        unique_ratio = nunique / n_rows if n_rows > 0 else 0

        should_convert = (savings > savings_threshold) or (unique_ratio < cardinality_threshold)

        detailed[col] = {
            'original_mem': orig_mem,
            'category_mem': cat_mem,
            'savings_ratio': savings,
            'nunique': nunique,
            'unique_ratio': unique_ratio,
            'should_convert': should_convert
        }

        if should_convert:
            suggested.append(col)

        if verbose:
            print(f"[{col}] mem_savings={savings:.2%}, nunique={nunique}, unique_ratio={unique_ratio:.2%}, convert={should_convert}")

    return detailed, suggested

def transform_and_analyze(
    df: pd.DataFrame,
    savings_threshold: float = 0.5,
    cardinality_threshold: float = 0.5,
    verbose: bool = True,
    output_path: str = None,
    output_format: str = 'csv'
):

    _, suggested = compare_memory_usage(
        df,
        savings_threshold=savings_threshold,
        cardinality_threshold=cardinality_threshold,
        verbose=False
    )

    new_df = df.copy()

    for col in suggested:
        new_df[col] = new_df[col].astype('category')
        if verbose:
            print(f"Converted '{col}' to category (nunique={new_df[col].nunique()})")

    stats = {}

    for col in new_df.select_dtypes(include=['category']).columns:
        stats[col] = list(new_df[col].cat.categories)

    numeric_df = new_df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        numeric_stats = numeric_df.describe().to_dict()
        stats.update(numeric_stats)

    saved_path = None
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'csv':
            new_df.to_csv(path, index=False)
        elif output_format == 'parquet':
            new_df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        saved_path = str(path.resolve())
        if verbose:
            print(f"\nSaved processed data to: {saved_path}")

    # if verbose:
    #     print("\n=== Conversion Summary ===")
    #     print(f"Converted columns: {suggested}")

    #     print("\n=== Statistics ===")
    #     for key, value in stats.items():
    #         if isinstance(value, dict):
    #             print(f"\n--- {key} (Numeric) ---")
    #             for stat, num in value.items():
    #                 print(f"  {stat}: {num:,.2f}")
    #         else:
    #             print(f"\n--- {key} (Category) ---")
    #             print(f"  Unique Values: {value}")

    return new_df, stats, saved_path


# def create_fact_extraction_prompt() -> ChatPromptTemplate:
#     system_template = (
#         "You are a chatbot that identifies and extracts essential factual terms from given input. "
#         "When provided with information, always respond in a JSON object format. "
#         "The object must include a list named 'key_facts' containing exactly "
#         "the 10 most relevant factual terms necessary to comprehend the input. "
#         "Do not provide any explanations."
#     )
    
#     user_template = "{input}"
    
#     return ChatPromptTemplate.from_messages([
#         ("system", system_template),
#         ("user", user_template)
#     ])

def create_fact_extraction_prompt() -> ChatPromptTemplate:
    system_template = (
        "You are a chatbot that extracts the most relevant factual terms from each field given as input.\n"
        "Input is always a JSON object with one or more fields, like: {{\"field\": \"some text\"}}\n"
        "Output must also be a JSON object with the **same field names**.\n"
        "For each field, return a list of extracted keywords.\n" 
        "**The number of keywords must not exceed 3 per field, and should depend on the length and relevance of the input text.**\n"
        "**Under no circumstances should the number of keywords exceed 3 per field.**\n"
        "Return only valid JSON, no explanations or additional text."
    )
    user_template = "{input}"
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def create_question_generate_prompt1() -> ChatPromptTemplate:

    system_template = (
        """
        You are a question contructor, a knowledge graph expert, and a linguist. You are skilled at raising unique questions from a given text and a specific relationship.
        """
    )
    user_template = (
        """
        # Task description
        Paraphrase this text as questions (up to 1) that tackle the content from various perspectives and can only be answered by reading the text. 
        The answer to the question be as lexically similar as possible to the original text, or even directly use the content in the original text. 
        Besides, The answers to these questions, when taken together, should comprehensively cover all the key information in the text. Avoid factual and simple yes/no questions. 
        Do not provide the answer, provide just the question. 
        In the last of each question, add a demanding sentence: "Do not answer with point form."

        # Input details:
        * `text`: A text that contains information about a specific relationship.

        # Input
        Text: {input}

        # Output
        A Json with key 'Question' and value being one of the questions you generated.   
        """
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def create_question_generate_prompt2() -> ChatPromptTemplate:

    system_template = (
        """
        You are an expert question constructor, knowledge graph specialist, and linguist. 
        You are skilled at generating precise, retrieval-oriented questions from a given text and entity, 
        ensuring the resulting questions are optimized for use in entity-aware Retrieval-Augmented Generation (RAG) systems.
        """
    )

    user_template = (
        """
        # Task Description  
        Paraphrase the provided text into a single question that addresses the content meaningfully and can only be answered by referring to the original text.  
        The answer should closely mirror the wording of the original text or directly quote parts of it.  
        The question must incorporate the provided entity to ensure effective retrieval in a RAG system where the text is associated with that entity.  
        Avoid factual or simple yes/no questions.  
        At the end of the question, include the instruction: "Do not answer with point form."

        # Input Details  
        - `Entity`: The main Entity to which the Text belongs.  
        - `Text`: A Text snippet containing information related to the entity.

        # Input  
        Entity: {Entity}  
        Text: {Text}  

        # Output  
        A JSON object with the key "Question" and the value as the generated question.  
        """
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def create_question_generate_prompt(version=0) -> ChatPromptTemplate:
    if version == 0:
        system_template = (
            """
            You are an expert question constructor specializing in vector database retrieval optimization. 
            Your task is to generate precise questions that will effectively retrieve the exact text snippet 
            from an entity-aware vector database when used as a query.
            
            Key principles for question generation:
            1. The question must be answerable ONLY by referencing the provided text
            2. It must incorporate the entity exactly as stored in the database
            3. The expected answer should closely mirror the original text wording
            4. Avoid factual, yes/no, or simple list questions
            5. Optimize for semantic matching in embedding-based retrieval systems
            """
        )

        user_template = (
            """
            # Vector Database Retrieval Context
            The vector database contains text chunks associated with specific entities.
            Your generated question will be used as a query to retrieve this exact text.
            
            # Task
            Create a question that:
            - Specifically targets the entity: {Entity}
            - Can only be answered by referencing this exact text snippet
            - Would effectively retrieve this text from the entity-aware vector database
            - Requires an answer that directly quotes or closely paraphrases the original text
            
            # Input
            Entity: {Entity}
            Text: {Text}
            
            # Output
            A JSON object with the key "Question" and the value as the generated question.
            Include the instruction: "Do not answer with point form." at the end.
            """
        )
        
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
    elif version == 1:
        pass


def create_question_generate_prompt4() -> ChatPromptTemplate:
    system_template = (
        "You are an expert at creating retrieval-optimized questions for vector databases. "
        "Generate precise questions that will retrieve exact text snippets when used as queries."
    )

    user_template = (
        """
        Create a question that retrieves the exact Text value for the specified Entity, using other columns as identifying conditions.

        Entity: {Entity}
        Columns: {Columns} (Text is one field value)
        Text: {Text} (The target value to retrieve)

        Guidelines:
        - Text is a field value within the Entity
        - Use other column values to uniquely identify the Entity
        - Ask specifically for the Text field value
        - Ensure the answer must be exactly the provided Text
        - Use natural language with multiple identifying conditions
        - End with: "Do not answer with point form."

        Example:
        Input: Entity="Alice", Columns="name, sex, age", Text="20"
        Output: {{"Question": "What is the age of the student named Alice who is female?"}}

        Generate a JSON object with key "Question" containing your question.
        """
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def get_ollama_llm(model_name: str = "llama3:8b", **kwargs) -> ChatOllama:
    defaults = {
        "temperature": 0.3,
        "top_p": 0.7,
        "timeout": 120
    }
    defaults.update(kwargs)
    return ChatOllama(model=model_name, **defaults)

def extract_idx(text):
    match = re.search(r'entity_(\d+)', text)
    if match:
        # return int(match.group(1))-1
        return int(match.group(1))
    else:
        return None

def extract_dataline(df, id):
    # a = df[df['id'] == id]
    row = df.loc[id].copy()
    content = '\n'.join([f"{col} : {row[col]}" for col in df.columns])
    return content

def create_formatted_questions(answer):

    question = f"Output the contents of the RAG dataset that most closely resemble the following text : {answer}"
    return question

# def create_RAGllm_chain(file_path, RAG_model="deepseek-r1:latest", emb_model="nomic-embed-text:latest", db_name="wine_db"):
#     # 读取parquet数据集
#     # RAG_df = pd.read_parquet(file_path, engine="auto")
#     # documents = dataframe_to_documents(RAG_df)

#     # 读取csv数据
#     csv_loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ','})
#     documents = csv_loader.load()

#     # 将文档分割为合适的块大小
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     embedding_model = OllamaEmbeddings(model=emb_model)
#     vector_db = Chroma.from_documents(texts, embedding_model, collection_name=db_name)

#     # 创建问答链
#     RAG_chain = RetrievalQA.from_chain_type(
#         llm = get_ollama_llm(model_name=RAG_model),
#         chain_type="stuff",
#         retriever=vector_db.as_retriever(),
#         return_source_documents=True
#     )
#     return RAG_chain

def dataframe_to_documents(df)->Document:
    documents = []
    for _, row in df.iterrows():
        # 将行中的列名和列值拼接成一个字符串
        content = ". ".join([f"{col}: {row[col]}" for col in df.columns])
        # 创建 Document 对象并添加到列表中
        documents.append(Document(page_content=content, metadata={"id": row['id'], "row_id": _}))
    return documents

def create_RAGdb(file_path, dataset_name, emb_model="bge-m3:latest", db_name="RAG_db"):

    vector_db_dir = f"vector_database/{dataset_name}"
    embedding_model = OllamaEmbeddings(model=emb_model)  # 提前创建嵌入模型

    if os.path.exists(vector_db_dir) and os.path.isdir(vector_db_dir):
        vector_db = Chroma(
            persist_directory=vector_db_dir,
            collection_name=db_name,
            embedding_function=embedding_model  # 加载时指定嵌入函数
        )
        print(f"Success load vector_db {dataset_name}_{db_name}")
        return vector_db

    # 如果不存在数据库或加载失败，则重新创建
    print(f"create vector_db {dataset_name}_{db_name}")
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

def create_answer_generate_prompt() -> PromptTemplate:
    combined_template = (
        """
        ### ROLE: 
        You are a factual answer generator using RAG datasets.

        ### TASK: 
        Answer questions strictly, using only the most relevant original content in the context provided in the RAG dataset.

        ### RULES:
        1. Do not provide any explanations.
        2. ALWAYS prioritize direct verbatim matches from the context
        3. If no matching content exists, respond ONLY with "No relevant information found"
        4. Extract the MOST relevant verbatim segment from the context
        5. NEVER rephrase, summarize or interpret the content
        6. Output ONLY the raw text excerpt with NO additional text

        ### CONTEXT:
        {context}

        ### QUESTION:
        {question}

        ### OUTPUT FORMAT:
        Raw verbatim text excerpt OR "No relevant information found"
        """
    )
    return PromptTemplate(
        template=combined_template,
        input_variables=["context", "question"]  # 必须包含context变量
    )

def create_RAGllm_chain(vector_db, RAG_model: str = "qwen:14b", source_documents=False, retrieval_k=5):
    """创建RAG问答链"""
    # 创建简单的问答提示模板
    answer_prompt = create_answer_generate_prompt()
    
    # 创建RAG链
    rag_chain = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(model_name=RAG_model),
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": retrieval_k}),  # 获取top5相关文档
        chain_type_kwargs={"prompt": answer_prompt},
        return_source_documents=source_documents
    )
    
    return rag_chain

def query_rag_system(chain, query):
    """使用RAG系统查询知识库"""
    result = chain.invoke({"query": query})
    return result['result']

def extract_keywords_light(row, field_names, chain):
    """轻量级关键词提取函数。

    Args:
        input_dict (dict): 输入字典，其中键是字段名，值是文本内容。

    Returns:
        dict: 输出字典，其中每个字段的值是一个包含关键词的列表。
    """
    input_dict = {
        field: str(row[field]) 
        for field in field_names 
        if pd.notnull(row[field]) and str(row[field]).strip()
    }
    if not input_dict:
        return {field: [] for field in field_names}
    stop_words = set(stopwords.words('english'))
    keywords_dict = {}

    for field, text in input_dict.items():
        # 文本预处理：移除标点符号并转为小写
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()

        # 分词
        words = word_tokenize(text)

        # 移除停用词并进行词性标注
        words = [word for word in words if word not in stop_words and word.isalpha()]
        pos_tags = pos_tag(words)

        # 提取名词作为关键词
        keywords = [word for word, tag in pos_tags if tag.startswith('NN')]

        # 去重并按出现频率排序
        keyword_freq = {}
        for word in keywords:
            keyword_freq[word] = keyword_freq.get(word, 0) + 1

        # 按频率排序并选择前3个关键词
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_keywords = [word for word, freq in sorted_keywords[:3]]

        keywords_dict[field] = sorted_keywords

    return keywords_dict


def extract_keywords(row, field_names, chain):
    """Extract keywords for a single row with retry logic"""
    
    # Prepare input, handling null values
    input_dict = {
        field: str(row[field]) 
        for field in field_names 
        if pd.notnull(row[field]) and str(row[field]).strip()
    }
    if not input_dict:
        return {field: [] for field in field_names}
    
    # Try extraction with retries
    try:
        response = chain.invoke({"input": input_dict})
        # clear_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL)
        # clean_response = remove_think_tags(response.content)
        return response
    
    except json.JSONDecodeError as e:
        print(f"---------Response content for row {row.name}:-------")
        print(f"{response.content}")
        print("--------------------response end---------------------")
        print(f"JSON parsing error for row {row.name} : {str(e)}")
    except Exception as e:
        print(f"---------Response content for row {row.name}:-------")
        print(f"{response.content}")
        print("--------------------response end---------------------")
        print(f"Error processing row {row.name} : {str(e)}")
    
    # Return empty lists if all attempts failed
    return {field: [] for field in field_names}

def create_single_field_rewrite_prompt() -> ChatPromptTemplate:
    # - Keywords list → Generate text incorporating these keywords
    system_template = """You are a data field rewriter. Your task is to generate a new value for a field based on auxiliary information.

RULES:
1. Output ONLY valid JSON: {{"field_name": "new_value"}}
2. Wrap your JSON response in <RESULT></RESULT> tags
3. Generate realistic values based on the auxiliary information provided
4. For numeric fields: generate ONLY numbers (integers or floats)
5. For categorical fields: select from provided categories when possible

AUXILIARY INFO TYPES:
- Values list → Select ONE value from the list exactly as written
- Statistics dict → Generate a number within min-max range, respect data type (int/float)

OUTPUT FORMAT:
<RESULT>
{{"field_name": "new_value"}}
</RESULT>"""

    user_template = """Field: {field_name}
Current: {current_value}
Auxiliary: {aux_info}

Generate new value:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])

def rewrite_single_field(field_name, current_value, aux_info, watermarked_llm):
    """Rewrite a single field using the watermarked LLM"""
    prompt = create_single_field_rewrite_prompt()
    chain = prompt | watermarked_llm
    
    try:
        result = chain.invoke({
            "field_name": field_name,
            "current_value": str(current_value),
            "aux_info": str(aux_info)
        })
        return result
    except Exception as e:
        print(f"Error rewriting field {field_name}: {e}")
        return str(current_value)


def load_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
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

def create_watermarked_llm(model, tokenizer, hash_key, max_new_tokens=128):
    """Create watermarked LLM with specific hash key"""
    try:
        watermarked_llm = KGWWatermarkedLLM(
            model=model,
            tokenizer=tokenizer,
            gamma=0.5,
            delta=2.0,
            hash_key=hash_key,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.eos_token_id,
            }
        )
        return watermarked_llm
        
    except Exception as e:
        print(f"Error creating watermarked LLM: {e}")
        return None
        
    except Exception as e:
        print(f"Error initializing watermarked LLM: {e}")
        return None

def extract_and_update_result(result_text: str, df: pd.DataFrame, row_idx: int, field: str) -> bool:
    pattern = r'<RESULT>\s*(.*?)\s*</RESULT>'
    match = re.search(pattern, result_text, re.DOTALL)
    
    if not match:
        print(f"No <RESULT> tag found in output for field {field}")
        return False
    
    result_content = match.group(1).strip()
    print(f"Extracted result for {field}: {result_content}")
    
    try:
        result_json = json.loads(result_content)
        
        if field in result_json:
            old_value = df.at[row_idx, field]
            new_value = result_json[field]
            if pd.api.types.is_categorical_dtype(df[field]):
                if str(new_value) not in df[field].cat.categories:
                    df[field] = df[field].cat.add_categories([str(new_value)])

            df.at[row_idx, field] = new_value
            print(f"Updated {field}: '{old_value}' -> '{new_value}'")
            return True
        else:
            print(f"Field {field} not found in result JSON")
            return False
            
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for field {field}: {e}")
        print(f"Raw content: {result_content}")
        return False

def find_dataset_max_tokens(df, column_name) -> int:
    """
    计算CSV文件中特定字段内容经tokenizer编码后的最大token长度
    
    参数:
        df: dataframe
        column_name: 需要分析的字段名
    
    返回:
        int: 最大token长度，如果字段不存在或文件为空则返回0
    """
    max_token_length = 0
    
    # 加载分词器（确保路径正确）
    try:
        tokenizer = AutoTokenizer.from_pretrained("../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    except Exception as e:
        print(f"Error: Failed to load tokenizer: {str(e)}")
        return 0

    try:
        
        # 检查字段是否存在
        if column_name not in df.columns:
            print(f"Error: Column name '{column_name}' not found in dataframe")
            return 0
        
        # 遍历所有行，计算每个字段内容的token长度
        row_count = 0
        for index, row in df.iterrows():
            row_count += 1
            # 获取字段内容并处理空值
            content = str(row[column_name]).strip()  # 确保转换为字符串
            if not content or content.lower() in ['nan', 'none']:
                continue  # 空内容无需计算
            
            # 编码并计算token长度（不添加特殊token）
            try:
                tokens = tokenizer.encode(
                    content,
                    add_special_tokens=False,
                    truncation=False  # 不截断，确保获取真实长度
                )
                current_length = len(tokens)
                
                # 更新最大长度
                if current_length > max_token_length:
                    max_token_length = current_length
            except Exception as e:
                print(f"处理第{row_count}行时编码出错: {str(e)}")
                continue
        
        print(f"Column {column_name} has {row_count} rows, max token length: {max_token_length}")
    
    except FileNotFoundError:
        print(f"Error: File not found")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return max_token_length

def find_dataset_min_tokens(df, column_name) -> int:
    """
    计算CSV文件中特定字段内容经tokenizer编码后的最小token长度
    
    参数:
        df: dataframe
        column_name: 需要分析的字段名
    
    返回:
        int: 最小token长度, 如果字段不存在或文件为空则返回0
    """
    min_token_length = 10000000000
    
    # 加载分词器（确保路径正确）
    try:
        tokenizer = AutoTokenizer.from_pretrained("../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    except Exception as e:
        print(f"Error: Failed to load tokenizer: {str(e)}")
        return 0

    try:
        
        # 检查字段是否存在
        if column_name not in df.columns:
            print(f"Error: Column name '{column_name}' not found in dataframe")
            return 0
        
        # 遍历所有行，计算每个字段内容的token长度
        row_count = 0
        for index, row in df.iterrows():
            row_count += 1
            # 获取字段内容并处理空值
            content = str(row[column_name]).strip()  # 确保转换为字符串
            if not content or content.lower() in ['nan', 'none']:
                continue  # 空内容无需计算
            
            # 编码并计算token长度（不添加特殊token）
            try:
                tokens = tokenizer.encode(
                    content,
                    add_special_tokens=False,
                    truncation=False  # 不截断，确保获取真实长度
                )
                current_length = len(tokens)
                
                # 更新最大长度
                if current_length < min_token_length:
                    min_token_length = current_length
            except Exception as e:
                print(f"处理第{row_count}行时编码出错: {str(e)}")
                continue
        
        print(f"Column {column_name} has {row_count} rows, min token length: {min_token_length}")
    
    except FileNotFoundError:
        print(f"Error: File not found")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return min_token_length

def to_original_path(csv_path: str):
    if "winemag50k" in csv_path:
        return "../dataset/winemag_sub_dataset_50k.csv", "winemag50k"
    elif "nfcorpus" in csv_path:
        return "../dataset/nfcorpus_corpus.csv", "nfcorpus"
    elif "FCT100k" in csv_path:
        return "../dataset/FCT_100k.csv", "FCT100k"
    elif "winemag5k" in csv_path:
        return "../dataset/winemag_sub_dataset_5k.csv", "winemag5k"
    else:
        return None, None
