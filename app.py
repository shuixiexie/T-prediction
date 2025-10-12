import streamlit as st
import pandas as pd
from compute_coupled_descriptors import compute_coupled_descriptors

# 设置网页配置
st.set_page_config(
    page_title="T-prediction",  # 浏览器标签页标题
    layout="wide"
)

# 网页显示标题
st.title("Mixture Descriptor Calculation")  # 网页顶部显示的大标题

# 上传 Excel 文件
uploaded_file = st.file_uploader(
    "上传包含 A/B/C 组件描述符的 Excel 文件", 
    type=["xlsx"]
)

if uploaded_file:
    st.info("正在验证文件和计算加权混合描述符，请稍等...")
    
    try:
        # 调用核心函数
        result_df = compute_coupled_descriptors(uploaded_file)
        
        st.success("计算完成！")
        
        # 显示结果
        st.dataframe(result_df)
        
        # 提供下载
        output_file = "Coupled_descriptors_result.xlsx"
        result_df.to_excel(output_file, index=False)
        st.download_button(
            label="下载结果 Excel",
            data=open(output_file, "rb"),
            file_name="Coupled_descriptors_result.xlsx"
        )
        
    except Exception as e:
        st.error(f"计算失败: {e}")




import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------
# 网页标题
# ---------------------------
st.title("Data Standardization")

# ---------------------------
# 上传原始 Excel 文件
# ---------------------------
uploaded_file = st.file_uploader("上传原始 Excel 文件进行标准化", type=["xlsx"])

if uploaded_file:
    try:
        st.info("正在读取数据...")
        df = pd.read_excel(uploaded_file)

        # ---------------------------
        # 1. 加载训练时保存的标准化器
        # ---------------------------
        scaler_dict = joblib.load("scalers_for_groups.save")  # 分组 StandardScaler
        group_cols = ["species", "endpoint", "effect", "Duration_Value"]  # 分组依据
        num_features = [
            "AATSC0v","nHBint2","minHCsats","FNSA-1","nHBint3",
            "RPSA","MATS5c","ETA_Epsilon_2","GATS4e","mindssC",
            "minHsOH","GATS4c","TDB5s","naaaC","minHBint5",
            "AATSC6p","MATS6c","ATSC4c","AATS5s","MATS6p"
        ]

        df_scaled_list = []

        # ---------------------------
        # 2. 按分组应用训练好的 scaler
        # ---------------------------
        grouped = df.groupby(group_cols)
        for group_name, group_data in grouped:
            if group_name in scaler_dict:
                scaler = scaler_dict[group_name]
                features_scaled = pd.DataFrame(
                    scaler.transform(group_data[num_features]),
                    columns=num_features,
                    index=group_data.index
                )
                group_scaled = pd.concat([group_data[group_cols], features_scaled], axis=1)
                df_scaled_list.append(group_scaled)
            else:
                st.warning(f"分组 {group_name} 未在训练集出现，未进行标准化")
                df_scaled_list.append(group_data)

        # 合并所有分组
        df_scaled = pd.concat(df_scaled_list, axis=0)

        st.success("标准化完成！")
        st.dataframe(df_scaled)

        # ---------------------------
        # 3. 提供下载
        # ---------------------------
        output_file = "standardized_data.xlsx"
        df_scaled.to_excel(output_file, index=False)
        with open(output_file, "rb") as f:
            st.download_button(
                label="下载标准化后的数据",
                data=f,
                file_name=output_file
            )

    except Exception as e:
        st.error(f"标准化失败: {e}")



import streamlit as st
import pandas as pd
import torch
import joblib
from transformer_model import TransformerMLP  # 你保存模型的类文件
import numpy as np

st.set_page_config(page_title="T-prediction", layout="wide")
st.title("Mixture Toxicity Prediction")

uploaded_file = st.file_uploader("上传待预测的 Excel 文件", type=["xlsx"])

if uploaded_file:
    try:
        st.info("正在读取文件...")

        # ---------------------------
        # 1. 读取用户上传的 Excel
        # ---------------------------
        predict_df = pd.read_excel(uploaded_file)

        # ---------------------------
        # 2. 读取训练时保存的特征列（顺序）和模型
        # ---------------------------
        feature_columns = joblib.load("feature_columns.save")  # 包含 one-hot 后列顺序
        categorical_cols = ["species", "endpoint", "effect", "Duration_Value"]  # 分类列

        # ---------------------------
        # 3. One-hot 编码 + 对齐列
        # ---------------------------
        predict_encoded = pd.get_dummies(predict_df, columns=categorical_cols)

        # 补全缺失列
        for col in feature_columns:
            if col not in predict_encoded.columns:
                predict_encoded[col] = 0

        # 重新排列列顺序
        predict_encoded = predict_encoded[feature_columns]

        # 确保全是数值类型
        predict_encoded = predict_encoded.astype(float)

        # ---------------------------
        # 4. 转换为 torch tensor
        # ---------------------------
        X_predict = torch.tensor(predict_encoded.values, dtype=torch.float32)

        # ---------------------------
        # 5. 加载模型
        # ---------------------------
        input_dim = X_predict.shape[1]
        model = TransformerMLP(input_dim=input_dim, hidden_dim=100, transformer_layers=5, num_heads=5, ff_dim=256, output_dim=1)
        model.load_state_dict(torch.load("transformer_DNN_model.pth", map_location=torch.device('cpu')))
        model.eval()

        # ---------------------------
        # 6. 预测
        # ---------------------------
        with torch.no_grad():
            predictions = model(X_predict).numpy().flatten()

        # ---------------------------
        # 7. 显示和下载预测结果
        # ---------------------------
        predict_df["Predicted"] = predictions
        st.success("预测完成！")
        st.dataframe(predict_df)

        output_file = "prediction_results.xlsx"
        predict_df.to_excel(output_file, index=False)
        st.download_button(
            label="下载预测结果",
            data=open(output_file, "rb"),
            file_name="prediction_results.xlsx"
        )

    except Exception as e:
        st.error(f"预测失败: {e}")
