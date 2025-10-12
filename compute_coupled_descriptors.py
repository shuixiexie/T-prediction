# app.py

import pandas as pd
import numpy as np
import joblib
import os
import uuid
import asyncio
import logging
from typing import List, Dict, Any

# descriptor libraries
from padelpy import from_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors as mordred_descriptors

import tempfile

from padelpy import padeldescriptor

selected_columns = [
    'nHBint3', 'nHBint2', 'minHCsats', 'FNSA-1', 'AATSC0v',
    "RPSA", "MATS5c", "ETA_Epsilon_2", "GATS4e", "mindssC",
    "minHsOH", "GATS4c", "TDB5s", "naaaC", "minHBint5",
    "AATSC6p", "MATS6c", "ATSC4c", "AATS5s", "MATS6p"
]




# -------------------
# 辅助函数
# -------------------
def compute_coupled_descriptors(file_path: str, output_path: str = None, tol: float = 1e-6) -> pd.DataFrame:
    # ------------------
    # 内部验证 Excel 文件
    # ------------------
    def validate_excel_file(file_path: str) -> bool:
        required_sheets = ['Component A descriptors', 'Component B descriptors', 'Component C descriptors']
        base_columns = ["species", "endpoint", "effect", "Duration_Value"]
        ratio_columns = ["A-ratio", "B-ratio", "C-ratio"]
        selected_columns = [
            "nHBint3", "nHBint2", "minHCsats", "FNSA-1", "AATSC0v",
            "RPSA", "MATS5c", "ETA_Epsilon_2", "GATS4e", "mindssC",
            "minHsOH", "GATS4c", "TDB5s", "naaaC", "minHBint5",
            "AATSC6p", "MATS6c", "ATSC4c", "AATS5s", "MATS6p"
        ]
        required_columns = base_columns + ratio_columns + selected_columns

        try:
            xls = pd.ExcelFile(file_path)
        except Exception as e:
            raise ValueError(f"无法读取 Excel 文件：{e}")

        missing_sheets = [s for s in required_sheets if s not in xls.sheet_names]
        if missing_sheets:
            raise ValueError(f"缺少以下活动表: {missing_sheets}")

        for sheet in required_sheets:
            df = pd.read_excel(xls, sheet_name=sheet)
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"活动表 [{sheet}] 缺少以下列: {missing_columns}")
            if len(df) == 0:
                raise ValueError(f"活动表 [{sheet}] 为空")
        return True

    validate_excel_file(file_path)

    # ------------------
    # 读取三个 sheet
    # ------------------
    sheet_names = ['Component A descriptors', 'Component B descriptors', 'Component C descriptors']
    data = {name: pd.read_excel(file_path, sheet_name=name) for name in sheet_names}

    # 填充比例列为空值为0
    for df in data.values():
        for ratio_col in ["A-ratio", "B-ratio", "C-ratio"]:
            df[ratio_col] = pd.to_numeric(df[ratio_col], errors='coerce').fillna(0)

    # 填充描述符 NaN 为 0
    for df in data.values():
        df.fillna(0, inplace=True)

    # 检查行数一致
    n_rows = len(data[sheet_names[0]])
    if not all(len(df) == n_rows for df in data.values()):
        raise ValueError("三个活动表的样本数量不一致，请检查输入文件。")

    # 检查比例列和是否为1
    ratio_sum = data[sheet_names[0]]["A-ratio"] + data[sheet_names[0]]["B-ratio"] + data[sheet_names[0]]["C-ratio"]
    if not np.all(np.abs(ratio_sum - 1) <= tol):
        raise ValueError("比例列不满足 A-ratio + B-ratio + C-ratio ≈ 1，请检查输入文件。")

    # ------------------
    # 构建输出 DataFrame（基础信息取自A表）
    # ------------------
    df_result = data[sheet_names[0]][["species", "endpoint", "Duration_Value", "effect"]].copy()
    A_ratio = data[sheet_names[0]]["A-ratio"]
    B_ratio = data[sheet_names[0]]["B-ratio"]
    C_ratio = data[sheet_names[0]]["C-ratio"]

    # ------------------
    # 固定输出列顺序
    # ------------------
    final_columns = [
        "AATSC0v","nHBint2","minHCsats","FNSA-1","nHBint3",
        "RPSA","MATS5c","ETA_Epsilon_2","GATS4e","mindssC",
        "minHsOH","GATS4c","TDB5s","naaaC","minHBint5",
        "AATSC6p","MATS6c","ATSC4c","AATS5s","MATS6p"
    ]

    # ------------------
    # 计算加权混合并直接覆盖列
    # ------------------
    for desc in final_columns:
        weighted_sum = (
            A_ratio * data[sheet_names[0]][desc] +
            B_ratio * data[sheet_names[1]][desc] +
            C_ratio * data[sheet_names[2]][desc]
        )
        df_result[desc] = weighted_sum

    # ------------------
    # 按固定顺序排列列
    # ------------------
    df_result = df_result[["species","endpoint","Duration_Value","effect"] + final_columns]

    # ------------------
    # 保存结果
    # ------------------
    if output_path:
        df_result.to_excel(output_path, index=False)
        print(f"✅ 已保存加权混合描述符结果至：{output_path}")

    return df_result






