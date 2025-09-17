import os
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from utils1 import hash_sig, select_ka_ca, hash_value_key, hash_int, select_OBJECT_fields

def test_watermark_distribution(df: pd.DataFrame, dataset_name, sig: str, K1: str, K2: str, L: int, gamma: int) -> dict:
    """测试水印比特分布情况（不执行实际重写）"""
    # 初始化统计字典
    stats = {
        'bit_distribution': [0] * L,          # 每个bit位的entity命中次数
        'text_field_hits': [0] * L,            # 每个bit位的文本字段命中次数
        'non_text_field_hits': [0] * L,        # 每个bit位的非文本字段命中次数
        'emb_count': [0] * L,                  # 每个bit位的总嵌入次数
        'total_entities_processed': 0,         # 处理的总实体数
        'entities_with_watermark': 0,          # 触发水印嵌入的实体数
        'original_bits': hash_sig(sig, L)      # 原始水印比特串
    }
    
    # 获取候选属性
    KA, A, CA_combos = select_ka_ca(df, dataset_name=dataset_name)
    candidate_attrs = CA_combos + KA
    
    # 识别文本字段和非文本字段
    # text_fields = [col for col in df.select_dtypes(include=['object']).columns if col in A]
    text_fields = select_OBJECT_fields(dataset_name=dataset_name)
    non_text_fields = [col for col in A if col not in text_fields]
    
    # 遍历数据集
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing distribution"):
        PA = None
        BP = None
        
        # 定位逻辑
        for I in candidate_attrs:
            val = str(row[I]) if isinstance(I, str) else "".join(str(row[f]) for f in I if f)
            if hash_value_key(val, K1, gamma) == 0:
                PA = I
                BP = hash_value_key(val, K2, L)
                break

        if not PA:
            continue
            
        stats['entities_with_watermark'] += 1
        stats['bit_distribution'][BP] += 1
        
        # 统计字段类型分布
        WA_fields = list(set(A) - set([PA] if isinstance(PA, str) else PA))
        
        for field in WA_fields:
            if field in text_fields:
                stats['text_field_hits'][BP] += 1
            else:
                stats['non_text_field_hits'][BP] += 1

        stats['emb_count'][BP]=stats['text_field_hits'][BP]+stats['non_text_field_hits'][BP]
    
    stats['total_entities_processed'] = len(df)
    
    return stats

def search_gamma_for_watermark_distribution(df, dataset_name, sig, K1, K2, L, rate, gamma_list, min_emb=-1, max_emb=1000000, save=False):
    """
    从给定的 gamma 列表中搜索满足条件的 gamma 值，并在这些 gamma 值中选取评估值最大的那个
    
    参数：
    df -- 数据集
    sig -- 水印签名
    K1 -- 密钥 1
    K2 -- 密钥 2
    L -- 水印长度
    rate -- 最小嵌入数/最大嵌入数
    gamma_list -- gamma 值列表
    data_size -- 数据集大小标识
    
    返回：
    评估值最大的 gamma 值及其对应的测试结果
    """
    valid_gammas = {}
    
    for gamma in tqdm(gamma_list, desc="Searching gamma"):
        test_stats = test_watermark_distribution(
            df=df,
            sig=sig,
            dataset_name=dataset_name,
            K1=K1,
            K2=K2,
            L=L,
            gamma=gamma
        )
        
        # 计算每个 bit 位的最大嵌入次数和最小嵌入次数
        max_hit = max(test_stats['bit_distribution'])
        min_hit = min(test_stats['bit_distribution'])
        
        # 判断每个 bit 位的嵌入次数是否满足条件
        # valid = all(hit >= rate * max_hit for hit in test_stats['bit_distribution']) and min_hit>5
        valid = min_hit>=min_emb and max_hit<=max_emb and sum(test_stats['bit_distribution']) < (rate * len(df))

        
        if valid:
            valid_gammas[gamma] = test_stats
    
    if valid_gammas:
        # 计算每个 gamma 值对应的评估值
        evaluations = {}
        for gamma, stats in valid_gammas.items():
            max_hit = max(stats['emb_count'])
            min_hit = min(stats['emb_count'])
            evaluation = max(test_stats['bit_distribution'])
            # evaluation = min_hit / max_hit
            evaluations[gamma] = evaluation
        
        # 找到评估值最大的 gamma 值
        best_gamma = max(evaluations, key=evaluations.get)
        best_stats = valid_gammas[best_gamma]
        
        # 保存搜索结果
        result_dir = "./data/watermark_distribution"
        os.makedirs(result_dir, exist_ok=True)
        result_filename = f"{result_dir}/{dataset_name}_g{best_gamma}.json"
        if save:
            with open(result_filename, "w") as f:
                json.dump({
                    'gamma': best_gamma,
                    'evaluation': evaluations[best_gamma],
                    'stats': best_stats
                }, f, indent=2)
        print(f"\nBest gamma search results saved to {result_filename}")
        
        # 打印评估值最大的 gamma 值及其对应的统计结果
        print("\n=== Best Gamma Search Results ===")
        print(f"Best Gamma: {best_gamma} (Evaluation: {evaluations[best_gamma]:.4f})")
        print(f"Entities Processed: {best_stats['total_entities_processed']}")
        print(f"Entities With Watermark: {best_stats['entities_with_watermark']} ({best_stats['entities_with_watermark']/best_stats['total_entities_processed']:.1%})")
        print("\nBit | Hit Count | Text Fields |  Non-Text Fields | emb_count")
        print("----|-----------|-------------|------------------|------------------")
        for bp in range(L):
            print(f"{bp:3d} | {best_stats['bit_distribution'][bp]:9d} | {best_stats['text_field_hits'][bp]:11d} | {best_stats['non_text_field_hits'][bp]:16d} | {best_stats['emb_count'][bp]:16d}")
        
        return best_gamma, best_stats
    else:
        print("\nNo valid gamma found in the given list.")
        return None, None

if __name__ == "__main__":

    gamma_list_0 = [587]

    gamma_list_1 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 
       83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

    gamma_list_2 = [
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 
        353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463]
    
    gamma_list_3 = [
        467, 479, 487, 491, 499, 503, 507, 521, 523, 541, 547, 563, 569, 571, 577, 587, 593, 599, 601, 607, ]
    
    gamma_list_4 = [613, 617, 619, 647, 653, 659, 683, 691, 701, 709, 719, 727, 733, 739]
    
    gamma_list_5 = [
        743, 751, 757, 809, 811, 821,
        911, 919, 929, 953, 967, 971,
    ]

    gamma_list_5 = [
        1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 
    ]


    gamma_list_6 = [
        1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709
    ]

    gamma_list_6 = [
        1721, 1831, 1847, 1861, 1867, 1871, 1873
    ]
    

    # 加载数据集
    # df = pd.read_csv(f"./data/winemag_sub_dataset_{data_size}.csv")
    # df = pd.read_csv("data/New_Medium_Data.csv")

    df = pd.read_csv("../dataset/FCT_100k.csv")
    # 运行超参搜索
    best_gamma, best_stats = search_gamma_for_watermark_distribution(
        df=df,
        dataset_name="FCT_100k",
        sig="watermark_test",
        K1="Key11",
        K2="Keyword2",
        L=16,
        rate=0.01,
        min_emb=3,
        # max_emb=50,
        gamma_list=gamma_list_6,
        save=True
    )
    
    # 打印最终结果
    if best_gamma is None:
        print("\nNo valid gamma found in the given list.")

    # 311
    # winemag5k 107
    # winemag5k 587