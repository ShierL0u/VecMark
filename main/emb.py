import os
import nltk
# 确保在导入torch之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
nltk.data.path.append('~/autodl-tmp/nltk_data')

import argparse
import pandas as pd
from utils1 import *
import secrets
import re
import json
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import sent_tokenize



def split_text_into_sentences(text):
    """Split text into sentences using NLTK"""
    try:
        # Download punkt if not already downloaded
        nltk.download('punkt', quiet=True)
        sentences = sent_tokenize(str(text))
        return sentences
    except:
        # Fallback to simple period splitting if NLTK fails
        sentences = re.split(r'[.!?]+', str(text))
        return [s.strip() for s in sentences if s.strip()]

def find_sentences_with_keywords(sentences, keywords):
    """Find sentences that contain any of the keywords"""
    sentences_with_keywords = []
    used_keywords = set()
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        sentence_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in sentence_lower and keyword not in used_keywords:
                sentence_keywords.append(keyword)
                used_keywords.add(keyword)
        
        if sentence_keywords:
            sentences_with_keywords.append({
                'index': i,
                'sentence': sentence,
                'keywords': sentence_keywords
            })
    
    return sentences_with_keywords

def rewrite_sentence_with_context(full_text, sentence_to_rewrite, keywords, watermarked_llm):
    """Rewrite a specific sentence within the context of the full text"""
    prompt = f"""Rewrite the following sentence to express the same meaning using different words, while keeping all the specified keywords unchanged.

Original sentence: {sentence_to_rewrite}

Required keywords to keep: {', '.join(keywords)}

Instructions:
- Make the new sentence at least 30 words long.
- Rewrite using different sentence structure or synonyms
- Keep all keywords: {', '.join(keywords)} exactly as they are
- Maintain the same meaning
- Return ONLY the single rewritten sentence
- Do not provide multiple versions or explanations

Rewritten sentence:"""
    
    try:
        response = watermarked_llm.invoke(prompt)
        # Extract the rewritten sentence
        rewritten = response.strip()
        
        # Clean up the response (remove quotes, extra formatting)
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        
        # Additional cleanup for common LLM response patterns
        lines = rewritten.split('\n')
        # Take the first non-empty line as the rewritten sentence
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Rewritten'):
                rewritten = line
                break
        
        return rewritten
    except Exception as e:
        print(f"Error rewriting sentence: {e}")
        return sentence_to_rewrite

def process_text_field_enhanced(field_name, text_content, keywords, watermarked_llm):
    """Enhanced text field processing with sentence-level rewriting"""
    if not text_content or not keywords:
        return text_content, {}
    
    # Split text into sentences
    sentences = split_text_into_sentences(text_content)
    
    # Find sentences containing keywords
    sentences_with_keywords = find_sentences_with_keywords(sentences, keywords)
    
    if not sentences_with_keywords:
        return text_content, {}
    
    # Create a copy of sentences to modify
    modified_sentences = sentences.copy()
    
    # Collect all keywords used in rewriting
    all_keywords_used = []
    
    # Process each sentence that contains keywords
    for sentence_info in sentences_with_keywords:
        idx = sentence_info['index']
        original_sentence = sentence_info['sentence']
        sentence_keywords = sentence_info['keywords']
        
        # Rewrite the sentence
        rewritten_sentence = rewrite_sentence_with_context(
            full_text=text_content,
            sentence_to_rewrite=original_sentence,
            keywords=sentence_keywords,
            watermarked_llm=watermarked_llm
        )
        
        # Update the sentence in the list
        modified_sentences[idx] = rewritten_sentence
        
        # Collect keywords used
        all_keywords_used.extend(sentence_keywords)
        
        print(f"Original: {original_sentence}")
        print(f"Rewritten: {rewritten_sentence}")
        print(f"Keywords: {sentence_keywords}")
        print("-" * 50)
    
    # Reconstruct the text
    reconstructed_text = ' '.join(modified_sentences)
    
    # Return the field-level mapping with all keywords and the complete rewritten text
    field_mapping = {
        'key_words': list(set(all_keywords_used)),  # Remove duplicates
        'overall_rewritten_sentence': reconstructed_text
    }
    
    return reconstructed_text, field_mapping

def save_rewrite_mapping(mapping_data, output_path="./data/rewrite_mapping.json"):
    """Save the rewrite mapping to a JSON file"""
    try:
        # Save new data directly (overwrite if file exists)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"Rewrite mapping saved to {output_path} (overwritten)")
    except Exception as e:
        print(f"Error saving rewrite mapping: {e}")

def embedding_watermark(df: pd.DataFrame, dataset_name:str, output_path: str, mapping_path, 
                        sig: str, K1: str, K2: str, L: int, gamma: int, onlyTEXT:bool=False) -> pd.DataFrame:
    
    wm_bits = hash_sig(sig, L)
    print("wm_bits: ", wm_bits)

    new_df, stats, _ = transform_and_analyze(
        df,
        savings_threshold=0.99,
        cardinality_threshold=0.001,
        verbose=True,
        output_path=f"./data/transformed_dataset/data_{dataset_name}.csv",
        output_format="csv"  
    )
    
    KA, A, CA_combos = select_ka_ca(df, dataset_name=dataset_name)
    print("================================================")
    Is = CA_combos + KA
    print("Is: ", Is)
    prompt = create_fact_extraction_prompt()
    llm = get_ollama_llm(model_name="deepseek-r1:latest")
    schema = create_json_schema(name=dataset_name)
    # llm = get_ollama_llm(model_name="qwen:14b").with_structured_output(schema)
    chain = prompt | llm

    model, tokenizer = load_model(model_name="../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    
    # Global rewrite mapping for all processed entities
    global_rewrite_mapping = {}
    
    # Track statistics
    processed_entities = 0
    successful_updates = 0
    
    for idx, row in tqdm(new_df.iterrows(), total=len(new_df), desc="Embedding watermark"):
    # for idx, row in new_df.iterrows():
        # print(f'Embedding watermark : {idx}/{len(new_df)}')
        PA = None
        BP = None

        for I in Is:
            if isinstance(I, str):
                val = str(row[I])
            elif isinstance(I, tuple):
                val = "".join(str(row[f]) for f in I if f)
            else:
                continue
            if hash_value_key(val, K1, gamma) == 0:
                PA = I
                BP = hash_value_key(val, K2, L)
                break

        if not PA: 
            continue

        processed_entities += 1

        bit = wm_bits[BP]
        hash_key = hash_int(K2, BP, bit, mod_num=11)
        print("PA: ", PA)
        WA_fields = list(set(A) - set(PA))
        # OBJECT_fields = [field for field in WA_fields if field in new_df.select_dtypes(include=['object']).columns]
        OBJECT_fields = select_OBJECT_fields(dataset_name=dataset_name)
        
        # keywords_dict = extract_keywords(row, OBJECT_fields, chain) 
        keywords_dict = extract_keywords_light(row, OBJECT_fields, chain) 
        for field, keywords in keywords_dict.items():
            print(f"{field}: {keywords}")

        stats_info = {field: stats[field] for field in WA_fields if field in stats.keys()}

        json_output = {field: (str(row[field])) for field in A if field in WA_fields}
        print("json_WA_fields: ", json.dumps(json_output, indent=2))

        watermarked_llm = create_watermarked_llm(model, tokenizer, hash_key)
        field_updates = 0
        
        # Entity-specific rewrite mapping
        entity_rewrite_mapping = {
            "bit_position": BP,  # 嵌入的比特位位置
            "bit_value": bit,    # 嵌入的比特值
        }
        
        for field in WA_fields:
            if field in OBJECT_fields and field in keywords_dict:
                # Enhanced text field processing
                original_text = str(row[field])
                keywords = keywords_dict[field]
                
                rewritten_text, field_rewrite_mapping = process_text_field_enhanced(
                    field_name=field,
                    text_content=original_text,
                    keywords=keywords,
                    watermarked_llm=watermarked_llm
                )
                # print(f"Rewritten result: {rewritten_text}")
                # print(f"field rewrite mapping: {field_rewrite_mapping}")
                
                # Update the dataframe
                new_df.at[idx, field] = rewritten_text
                
                # Store rewrite mapping in the new format
                if field_rewrite_mapping:
                    entity_rewrite_mapping[field] = field_rewrite_mapping
                    field_updates += 1
                    successful_updates += 1
                    print(f"Successfully updated field {field} in row {idx}")
                else:
                    print(f"No rewriting needed for field {field} in row {idx}")
                    
            elif field in stats_info and not onlyTEXT:
                # Handle numeric/categorical fields (existing logic)
                aux_info = stats_info[field]
                result = rewrite_single_field(
                    field_name=field,
                    current_value=row[field],
                    aux_info=aux_info,
                    watermarked_llm=watermarked_llm
                )
                # print(f"Rewritten result: {result}")
                
                success = extract_and_update_result(result, new_df, idx, field)
                if success:
                    print(f"Successfully updated field {field} in row {idx}")
                    field_updates += 1
                    successful_updates += 1
                else:
                    print(f"Failed to update field {field} in row {idx}")
        
        # Add entity mapping to global mapping in the new format
        if entity_rewrite_mapping:
            entity_key = f"entity_{idx}"
            # print(f"add entity_key {entity_key}")
            global_rewrite_mapping[entity_key] = entity_rewrite_mapping
        
        print(f"Updated {field_updates} fields for entity {idx}")
        print("================================================")
        
        # Optional: Add a break for testing with limited entities
        # if processed_entities >= 10:
        #     break

    # # 在 embedding_watermark 函数中，保存前添加：
    # for col in new_df.columns:
    #     # 尝试将看起来像数字的列转换为数值类型
    #     try:
    #         new_df[col] = pd.to_numeric(new_df[col], errors='ignore')
    #     except:
    #         pass

    # # 然后再保存
    new_df.to_csv(output_path, index=False)

    # new_df.to_parquet(output_path, index=False)
    print(f"Watermarked data saved to {output_path}")
    
    # Save rewrite mapping
    save_rewrite_mapping(global_rewrite_mapping, output_path=mapping_path)
    
    print(f"Processed {processed_entities} entities, successfully updated {successful_updates} fields")
    print(f"Total entities with rewrite mappings: {len(global_rewrite_mapping)}")
    
    return new_df

if __name__ == "__main__":

    # df = pd.read_csv("./data/winemag_sub_dataset_500.csv")
    # embedding_watermark(df=df,
    #     output_path="./data/watermarked_data_500.parquet",
    #     sig="watermark_test",
    #     K1="Key11",
    #     K2="Keyword2",
    #     L=16,
    #     gamma=463)
    dataset_name = "FCT100k"
    if dataset_name == "nfcorpus":
        gamma = 37
        df = pd.read_csv("dataset/nfcorpus_corpus.csv")
    elif dataset_name == "winemag5k":
        gamma = 463
        df = pd.read_csv("dataset/winemag_sub_dataset_5k.csv")
    elif dataset_name == "winemag50k":
        gamma = 587
        df = pd.read_csv("dataset/winemag_sub_dataset_50k.csv")
    elif dataset_name == "FCT":
        gamma = 5059
        df = pd.read_csv("../dataset/FCT.csv")
    elif dataset_name == "FCT100k":
        gamma = 1721
        df = pd.read_csv("../dataset/FCT_100k.csv")
    
    # df = pd.read_csv(f"./data/winemag_sub_dataset_{data_size}.csv")
    # df = pd.read_csv("data/New_Medium_Data.csv")
    
    embedding_watermark(df=df,
        dataset_name=dataset_name,
        output_path=f"./data/watermarked_data/watermarked_{dataset_name}_g{gamma}.csv",
        mapping_path=f"data/mapping/{dataset_name}_g{gamma}_rewrite_mapping.json",
        sig="watermark_test",
        K1="Key11",
        K2="Keyword2",
        L=16,
        gamma=gamma)
