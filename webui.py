import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

# --- 导入你的自定义模块 ---
# 确保这些 .py 文件在同一目录下
try:
    import config as default_config
    from extract_content import ContentExtractor
    from data_sanitizer import TaxDataSanitizer
    from data_analysis import TaxClusteringEngine
    from cluster_tagger import LLMKeywordExtractor
    from excel_handle import read_excel, save_excel
except ImportError as e:
    st.error(f"❌ 模块导入失败: {e}")
    st.stop()

# ==========================================
# 0. 全局配置与状态管理
# ==========================================
st.set_page_config(page_title="税务数据智能分析平台", layout="wide", page_icon="📊")

CONFIG_FILE = "app_config.json"

def load_config():
    """加载配置：优先读取 JSON，否则读取 config.py，否则使用默认值"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 从 config.py 读取默认值
        return {
            "EXCEL_PATH": getattr(default_config, 'EXCEL_PATH', 'data.xlsx'),
            "SAVE_PATH": getattr(default_config, 'SAVE_PATH', 'processed_data.xlsx'),
            "STATE_FILE": getattr(default_config, 'STATE_FILE', 'cluster_state.pkl'),
            "EMBEDDING_MODEL": getattr(default_config, 'EMBEDDING_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2'),
            "LLM_API_KEY": "sk-xxxxxxxx",
            "LLM_BASE_URL": "https://api.deepseek.com"
        }

def save_config(cfg):
    """保存配置到 JSON"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    st.toast("✅ 配置已保存！")

# 初始化 Session State
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = None
if 'cluster_engine' not in st.session_state:
    st.session_state.cluster_engine = None

# ==========================================
# 1. 侧边栏：配置中心
# ==========================================
with st.sidebar:
    st.title("⚙️ 系统配置")
    
    with st.expander("📁 文件路径设置", expanded=True):
        st.session_state.config['EXCEL_PATH'] = st.text_input("原始 Excel 路径", st.session_state.config['EXCEL_PATH'])
        st.session_state.config['SAVE_PATH'] = st.text_input("预处理保存路径", st.session_state.config['SAVE_PATH'])
        st.session_state.config['STATE_FILE'] = st.text_input("聚类状态文件 (.pkl)", st.session_state.config['STATE_FILE'])

    with st.expander("🤖 模型与 API 设置", expanded=True):
        st.session_state.config['EMBEDDING_MODEL'] = st.text_input("SBERT 模型路径", st.session_state.config['EMBEDDING_MODEL'])
        st.session_state.config['LLM_API_KEY'] = st.text_input("LLM API Key", st.session_state.config['LLM_API_KEY'], type="password")
        st.session_state.config['LLM_BASE_URL'] = st.text_input("LLM Base URL", st.session_state.config['LLM_BASE_URL'])

    if st.button("💾 保存配置到文件"):
        save_config(st.session_state.config)

# ==========================================
# 2. 主界面：分步工作流
# ==========================================
st.title("📊 税务诉求聚类分析系统")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 数据清洗", "2️⃣ 聚类分析", "3️⃣ AI 深度洞察", "4️⃣ 结果导出"])

# --- Tab 1: 数据清洗与脱敏 ---
with tab1:
    st.header("数据预处理流水线")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"当前读取文件: `{st.session_state.config['EXCEL_PATH']}`")
    with col2:
        if st.button("🔄 执行清洗任务", type="primary"):
            try:
                with st.status("正在执行处理流程...", expanded=True) as status:
                    # 1. 读取
                    status.write("📂 读取原始 Excel...")
                    df = read_excel(st.session_state.config['EXCEL_PATH'], sheet_name=0, header=0)
                    
                    # 2. 生成 Trace_ID
                    status.write("🆔 生成 Trace_ID...")
                    df['Trace_ID'] = df.index + 2
                    
                    # 3. 提取内容
                    status.write("🧹 提取正文并去除废话...")
                    extractor = ContentExtractor()
                    # 假设 config 里有列名配置，或者默认 '业务内容'
                    col_name = '业务内容' if '业务内容' in df.columns else df.columns[0]
                    raw_content = extractor.extract_content(df, col_name)
                    
                    # 4. 脱敏
                    status.write("🛡️ 执行隐私脱敏 (NER + Regex)...")
                    sanitizer = TaxDataSanitizer(use_ner=True)
                    temp_df = pd.DataFrame({'text': raw_content})
                    sanitizer.process_dataframe(temp_df, 'text')
                    
                    # 5. 组装结果
                    clean_df = pd.DataFrame({
                        'Trace_ID': df['Trace_ID'],
                        'Original_Content': df[col_name],
                        'Extracted_Content': temp_df['text_sanitized']
                    })
                    
                    # 6. 过滤空行
                    clean_df = clean_df[clean_df['Extracted_Content'].str.strip() != '']
                    
                    # 保存到 Session 和 文件
                    st.session_state.clean_df = clean_df
                    st.session_state.raw_df = df # 保存原始完整数据供后续合并
                    
                    save_excel(clean_df, st.session_state.config['SAVE_PATH'])
                    status.update(label="✅ 预处理完成！", state="complete", expanded=False)
                    
                st.success(f"成功处理 {len(clean_df)} 条数据，已保存至 {st.session_state.config['SAVE_PATH']}")
                
            except Exception as e:
                st.error(f"处理失败: {e}")

    # 展示数据预览
    if st.session_state.clean_df is not None:
        st.subheader("数据预览")
        st.dataframe(st.session_state.clean_df.head())

# --- Tab 2: 聚类分析 ---
with tab2:
    st.header("核心聚类引擎")
    
    # 尝试自动加载已处理的数据
    if st.session_state.clean_df is None and os.path.exists(st.session_state.config['SAVE_PATH']):
        try:
            st.session_state.clean_df = read_excel(st.session_state.config['SAVE_PATH'])
            st.info("已自动加载上次处理的清洗数据。")
        except:
            pass

    if st.session_state.clean_df is None:
        st.warning("⚠️ 请先在 Tab 1 完成数据清洗")
    else:
        # 参数控制区
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_neighbors = st.slider("UMAP Neighbors", 5, 50, 15)
        with c2:
            min_cluster_size = st.slider("Min Cluster Size", 3, 100, 10)
        with c3:
            threshold = st.slider("合并相似度阈值", 0.8, 1.0, 0.92)
        with c4:
            load_mode = st.radio("模式", ["加载存档 (快)", "重新运行 (慢)"], index=0)

        if st.button("🚀 运行聚类分析", type="primary"):
            with st.spinner("正在运算中..."):
                # 初始化引擎
                if st.session_state.cluster_engine is None:
                    st.session_state.cluster_engine = TaxClusteringEngine(st.session_state.config['EMBEDDING_MODEL'])
                
                engine = st.session_state.cluster_engine
                
                # 准备数据
                texts = st.session_state.clean_df['Extracted_Content'].astype(str).tolist()
                ids = st.session_state.clean_df['Trace_ID'].tolist()
                
                # 执行逻辑
                if load_mode == "加载存档 (快)" and engine.load_state(st.session_state.config['STATE_FILE']):
                    st.success("成功加载历史状态！")
                else:
                    engine.run_analysis(texts, ids=ids, n_neighbors=n_neighbors, min_cluster_size=min_cluster_size)
                    engine.save_state(st.session_state.config['STATE_FILE'])
                    st.success("重新计算完成！")
                
                # 自动合并
                if threshold < 1.0:
                    engine.merge_similar_clusters(threshold=threshold)

        # 结果可视化
        if st.session_state.cluster_engine and st.session_state.cluster_engine.results_df is not None:
            res_df = st.session_state.cluster_engine.results_df
            
            # 统计指标
            n_clusters = res_df['Cluster'].nunique() - (1 if -1 in res_df['Cluster'].values else 0)
            st.metric("聚类话题数量", n_clusters)
            
            # Sunburst 图
            fig = px.sunburst(
                res_df[res_df['Cluster'] != -1], 
                path=['Keywords', 'Cluster'], 
                title="话题层级分布图"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("查看聚类结果表"):
                st.dataframe(res_df)

# --- Tab 3: AI 深度洞察 ---
with tab3:
    st.header("AI 语义分析实验室")
    
    engine = st.session_state.cluster_engine
    
    if engine is None or engine.results_df is None:
        st.warning("⚠️ 请先在 Tab 2 完成聚类")
    else:
        col_llm, col_sub = st.columns(2)
        
        # 1. LLM 关键词提取
        with col_llm:
            st.subheader("🤖 全局关键词提取")
            st.markdown("利用 LLM 对每个聚类进行精准概括。")
            if st.button("开始 AI 标注"):
                if not st.session_state.config['LLM_API_KEY']:
                    st.error("请先在左侧侧边栏配置 API Key")
                else:
                    with st.spinner("正在调用 LLM API..."):
                        engine.apply_llm_keywords(
                            api_key=st.session_state.config['LLM_API_KEY'],
                            base_url=st.session_state.config['LLM_BASE_URL']
                        )
                        st.success("标注完成！")
                        st.dataframe(engine.results_df[['Cluster', 'LLM_Keywords']].drop_duplicates().head())

        # 2. 子聚类显微镜
        with col_sub:
            st.subheader("🔬 子聚类显微镜")
            st.markdown("选择一个大类，挖掘其内部的细微差异。")
            
            # 选择器
            clusters = sorted(engine.results_df['Cluster'].unique())
            selected_cluster = st.selectbox("选择要钻取的大类 ID", [c for c in clusters if c != -1])
            
            if st.button("钻取分析"):
                if hasattr(engine, 'analyze_sub_cluster'):
                    sub_df = engine.analyze_sub_cluster(parent_cluster_id=selected_cluster, min_sub_size=3)
                    st.write(f"在 Cluster {selected_cluster} 中发现了 {sub_df['Sub_Cluster'].nunique()} 个子问题：")
                    
                    # 展示子聚类详情
                    for sub_id in sorted(sub_df['Sub_Cluster'].unique()):
                        if sub_id == -1: continue
                        with st.expander(f"子问题 #{sub_id} (样本数: {len(sub_df[sub_df['Sub_Cluster']==sub_id])})"):
                            st.write(sub_df[sub_df['Sub_Cluster']==sub_id]['Text'].head(5).tolist())
                else:
                    st.error("你的聚类引擎似乎缺少 `analyze_sub_cluster` 方法，请检查代码。")

# --- Tab 4: 结果导出 ---
with tab4:
    st.header("报表生成与导出")
    
    engine = st.session_state.cluster_engine
    
    if engine and engine.results_df is not None:
        st.markdown("系统将根据 `Trace_ID` 自动关联聚类结果与原始数据。")
        
        if st.button("生成最终报表"):
            # 1. 准备聚类结果
            cluster_res = engine.results_df
            
            # 2. 读取原始数据 (如果 Session 里没有，就去读文件)
            if st.session_state.raw_df is None:
                st.session_state.raw_df = read_excel(st.session_state.config['EXCEL_PATH'])
                st.session_state.raw_df['Trace_ID'] = st.session_state.raw_df.index + 2
            
            raw_df = st.session_state.raw_df
            
            # 3. 合并
            final_df = pd.merge(
                cluster_res,
                raw_df,
                on='Trace_ID',
                how='left',
                suffixes=('_Cluster', '_Original')
            )
            
            # 4. 整理列 (把重要的放前面)
            cols = ['Trace_ID', 'Cluster', 'LLM_Keywords', 'Keywords', 'Extracted_Content', 'Original_Content']
            # 动态调整，因为列名可能不一样
            existing_cols = [c for c in cols if c in final_df.columns]
            remaining_cols = [c for c in final_df.columns if c not in existing_cols]
            final_df = final_df[existing_cols + remaining_cols]
            
            st.dataframe(final_df.head())
            
            # 5. 下载按钮
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 下载最终 Excel 报表",
                data=output.getvalue(),
                file_name="税务聚类分析最终报告.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("暂无结果，请先完成前序步骤。")