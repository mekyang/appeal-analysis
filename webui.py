import streamlit as st
import pandas as pd
import os
import time

# 导入配置
from config import *

# 导入自定义模块
from extract_content import ContentExtractor_12366, ContentExtractor_12345
from data_sanitizer import TaxDataSanitizer
from data_analysis import TaxClusteringEngine 
from cluster_eval import ClusterEvaluator
from cluster_tagger import LLMKeywordExtractor
from excel_handle import *

# =====================================================================
# 页面配置
# =====================================================================
st.set_page_config(
    page_title="税务诉求分析系统 Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 侧边栏导航
# =====================================================================
st.sidebar.title("📋 导航菜单")
page = st.sidebar.radio(
    "选择功能模块",
    ["🏠 首页", "📤 数据预处理", "🧠 文本聚类分析", "📊 聚类评估", "🏷️ LLM关键词提取", "📈 结果查看"]
)

# =====================================================================
# 首页
# =====================================================================
if page == "🏠 首页":
    st.title("🎯 税务诉求分析系统")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("系统版本", "v2.0", "Pro")
    with col2:
        st.metric("处理模型", "SBERT + HDBSCAN", "生产级")
    with col3:
        st.metric("状态", "✅ 就绪", "可用")
    
    st.markdown("---")
    st.subheader("📚 功能介绍")
    
    features = {
        "📤 数据预处理": "✓ 内容提取 ✓ 隐私脱敏 ✓ 数据清洗",
        "🧠 文本聚类": "✓ SBERT编码 ✓ 状态保存 ✓ 相似簇合并",
        "📊 聚类评估": "✓ 轮廓系数 ✓ 散点图可视化 ✓ 相似度分析",
        "🏷️ LLM提取": "✓ DeepSeek/GPT调用 ✓ 关键词归纳 ✓ 智能采样",
        "📈 结果导出": "✓ Excel汇总 ✓ 统计分析 ✓ 报表生成"
    }
    
    for title, desc in features.items():
        st.success(f"**{title}**: {desc}")
    
    st.info("💡 建议流程: 数据预处理 → 文本聚类 (保存状态) → 聚类评估 → LLM提取 → 结果查看")

# =====================================================================
# 数据预处理页面 (修复版：解决按钮无法点击问题)
# =====================================================================
elif page == "📤 数据预处理":
    st.title("📤 数据预处理模块")
    st.markdown("步骤: 内容提取 → 隐私脱敏 → 保存结果")
    st.markdown("---")
    
    # --- 1. 初始化 Session State (关键步骤) ---
    # 用来“记住”处理后的数据，防止页面刷新后数据丢失
    if 'proc_df' not in st.session_state:
        st.session_state['proc_df'] = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ 文件上传")
        uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"])
        
        extractor_type = st.selectbox(
            "选择提取器类型",
            ["12366工单提取器", "电话记录简易提取器"]
        )
    
    with col2:
        st.subheader("2️⃣ 处理配置")
        use_ner = st.checkbox("启用NER模型脱敏", value=True, help="使用BERT模型识别公司名（更准确但更慢）")
        column_name = st.text_input("输入内容列名", "业务内容")
    
    st.markdown("---")
    
    # --- 2. 处理逻辑 (点击后只负责处理数据并存入 State) ---
    if st.button("🚀 开始处理", key="preprocess_btn"):
        if uploaded_file is None:
            st.error("❌ 请先上传Excel文件")
        else:
            try:
                # 读取文件
                with st.spinner("📖 正在读取文件..."):
                    df = pd.read_excel(uploaded_file)
                    st.success(f"✅ 文件加载成功，共 {len(df)} 行")
                
                # 内容提取
                with st.spinner("🔍 正在提取内容..."):
                    if extractor_type == "12366工单提取器":
                        extractor = ContentExtractor_12366()
                    else:
                        extractor = ContentExtractor_12345()
                    
                    df['Extracted_Content'] = extractor.extract_content(df, column_name)
                
                # 隐私脱敏
                if use_ner:
                    with st.spinner("🔐 正在进行隐私脱敏..."):
                        sanitizer = TaxDataSanitizer(use_ner=use_ner)
                        df = sanitizer.process_dataframe(df, 'Extracted_Content', 'Sanitized_Content')
                else:
                    df['Sanitized_Content'] = df['Extracted_Content']

                # 数据清洗
                initial_count = len(df)
                df = df[df['Sanitized_Content'].str.strip() != '']
                final_count = len(df)
                
                st.info(f"🧹 已过滤空行: {initial_count} → {final_count} (删除 {initial_count - final_count} 行)")
                
                # 【关键】将处理好的 df 存入 session_state
                st.session_state['proc_df'] = df
                st.success("✅ 处理完成！结果已缓存，请在下方查看和保存。")
                
            except Exception as e:
                st.error(f"❌ 处理失败: {str(e)}")

    # --- 3. 结果展示与保存逻辑 (独立于按钮之外) ---
    # 只要 State 里有数据，就显示这部分界面
    if st.session_state['proc_df'] is not None:
        df = st.session_state['proc_df']
        
        st.markdown("---")
        st.subheader("📋 预览处理结果")
        
        # 预览
        preview_cols = st.multiselect(
            "选择显示列",
            df.columns.tolist(),
            default=['Extracted_Content', 'Sanitized_Content']
        )
        st.dataframe(df[preview_cols].head(10), use_container_width=True)
        
        # 保存区域
        st.subheader("💾 保存结果")
        output_filename = st.text_input("输出文件名", "processed_data.xlsx")
        
        # 这个保存按钮现在在主层级，点击它不会导致数据丢失
        if st.button("💾 保存为Excel"):
            try:
                # 修正保存目录
                save_dir = os.path.dirname(OUTPUT_DIR)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                final_save_path = os.path.join(save_dir, output_filename)
                
                try:
                    save_excel(df, final_save_path)
                    st.success(f"✅ 保存成功！")
                    st.markdown(f"**文件位置:** `{final_save_path}`")
                    
                    # 仅限Windows本地打开文件夹
                    if st.button("📂 打开所在文件夹"):
                        try:
                            os.startfile(save_dir)
                        except:
                            st.warning("无法自动打开文件夹，请手动查看。")
                        
                except PermissionError:
                    st.error(f"❌ 保存失败：文件 `{output_filename}` 正被打开，请关闭后重试！")
                except Exception as e:
                    st.error(f"❌ 保存出错: {str(e)}")

            except Exception as e:
                st.error(f"❌ 路径配置错误: {str(e)}")
            
        # 下载按钮
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载为CSV",
            data=csv_buffer,
            file_name=output_filename.replace('.xlsx', '.csv'),
            mime="text/csv"
        )
# =====================================================================
# 🧠 文本聚类分析页面 (修复版V3 - 已整合)
# =====================================================================
elif page == "🧠 文本聚类分析":
    st.title("🧠 文本聚类分析 (高级版)")
    st.markdown("支持 **SBERT编码** + **参数微调** + **状态恢复** + **语义合并**")
    st.markdown("---")

    # --- Session State 初始化 ---
    if 'cluster_engine' not in st.session_state:
        st.session_state['cluster_engine'] = None

    # ==================================================
    # 第一部分：数据加载与任务创建
    # ==================================================
    with st.expander("📂 1. 数据源配置 (开始新任务或加载历史)", expanded=True):
        col_src1, col_src2 = st.columns(2)
        
        # --- 左侧：加载历史 ---
        with col_src1:
            st.subheader("方式 A: 加载历史状态")
            if os.path.exists(STATE_FILE):
                st.success(f"✅ 检测到存档: `{os.path.basename(STATE_FILE)}`")
                if st.button("📂 加载历史状态 (跳过漫长编码)", key="load_state_btn"):
                    try:
                        with st.spinner("正在恢复向量数据与聚类结果..."):
                            engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
                            if engine.load_state(STATE_FILE):
                                st.session_state['cluster_engine'] = engine
                                st.success(f"已恢复状态，包含 {len(engine.get_results())} 条数据")
                                time.sleep(0.5)
                                st.rerun()
                    except Exception as e:
                        st.error(f"加载失败: {str(e)}")
            else:
                st.warning("⚠️ 未找到历史状态文件 (首次运行请使用右侧)")

        # --- 右侧：上传新文件 ---
        with col_src2:
            st.subheader("方式 B: 上传新文件 (重新跑)")
            input_file = st.file_uploader("上传预处理后的Excel", type=["xlsx", "xls"], key="cluster_input")
            text_column = st.text_input("文本列名", "Sanitized_Content")

    # 如果上传了新文件，显示“开始运行”的配置面板
    if input_file is not None:
        st.info("检测到新文件上传，请配置参数并开始分析：")
        with st.form("new_analysis_form"):
            c1, c2, c3, c4 = st.columns(4)
            with c1: n_neighbors = st.slider("n_neighbors", 5, 50, 15)
            with c2: n_components = st.slider("n_components", 2, 10, 5)
            with c3: min_cluster_size = st.slider("min_cluster_size", 3, 100, 10)
            with c4: keyword_top_n = st.number_input("关键词数", 3, 10, 5)
            
            # 保存选项
            auto_save_new = st.checkbox("运行完成后自动保存状态文件", value=True)
            
            submitted = st.form_submit_button("🚀 开始完整分析 (SBERT编码)")
            
            if submitted:
                try:
                    df = pd.read_excel(input_file)
                    if text_column not in df.columns:
                        st.error(f"❌ 列名 '{text_column}' 不存在！")
                    else:
                        texts = df[text_column].dropna().astype(str).tolist()
                        texts = [t for t in texts if len(t.strip()) > 0]
                        
                        # 尝试获取ID
                        original_ids = None
                        for col in ['Trace_ID', '工单编号', 'ID', '序号']:
                            if col in df.columns:
                                original_ids = df[col].tolist()
                                break

                        with st.spinner("正在进行 SBERT 编码与聚类 (可能需要几分钟)..."):
                            engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
                            engine.run_analysis(
                                texts, 
                                original=original_ids,
                                n_neighbors=n_neighbors,
                                n_components=n_components,
                                min_cluster_size=min_cluster_size,
                                keyword_top_n=keyword_top_n
                            )
                            st.session_state['cluster_engine'] = engine
                            
                            if auto_save_new:
                                engine.save_state(STATE_FILE)
                                st.success("✅ 分析完成并已保存状态！")
                            else:
                                st.success("✅ 分析完成 (未保存状态)！")
                            
                            time.sleep(1)
                            st.rerun()
                except Exception as e:
                    st.error(f"分析失败: {str(e)}")

    st.markdown("---")

    # ==================================================
    # 第二部分：核心操作台 (仅在有数据时显示)
    # ==================================================
    if st.session_state['cluster_engine'] is not None:
        engine = st.session_state['cluster_engine']
        
        # --- 全局控制栏 ---
        st.subheader("🛠️ 2. 聚类控制台")
        
        # 放置一个全局开关，控制是否自动保存
        col_opt1, col_opt2 = st.columns([1, 3])
        with col_opt1:
            enable_auto_save = st.toggle("操作后自动更新状态文件", value=True, help="开启后，每次重聚类或合并都会覆盖 STATE_FILE")
        with col_opt2:
            if st.button("💾 手动保存当前状态"):
                engine.save_state(STATE_FILE)
                st.toast("✅ 状态已手动保存！")

        # --- 选项卡操作 ---
        tab1, tab2, tab3, tab4 = st.tabs(["🔄 快速重聚类", "🧩 合并相似簇", "📊 结果预览", "📤 导出数据"])
        
        # === Tab 1: 快速重聚类 ===
        with tab1:
            st.markdown("##### 调整密度参数 (无需重跑向量化)")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                re_min_size = st.slider("新 min_cluster_size", 3, 100, 10, key="re_min")
            with col_r2:
                re_neighbors = st.slider("新 n_neighbors", 5, 50, 15, key="re_neigh")
            with col_r3:
                re_top_n = st.number_input("新 关键词数", 1, 10, 5, key="re_top")
                
            if st.button("⚡ 执行重聚类", type="primary"):
                with st.spinner("正在重聚类..."):
                    engine.re_cluster(n_neighbors=re_neighbors, min_cluster_size=re_min_size, keyword_top_n=re_top_n)
                    
                    if enable_auto_save:
                        engine.save_state(STATE_FILE)
                        st.success("✅ 重聚类完成 (状态已自动保存)")
                    else:
                        st.success("✅ 重聚类完成 (仅内存更新)")
                    
                    time.sleep(0.5)
                    st.rerun()

        # === Tab 2: 合并相似簇 ===
        with tab2:
            st.markdown("##### 语义合并")
            st.info("功能说明：基于余弦相似度，将语义非常接近的簇合并为一个（例如：'开发票' 和 '开具发票'）。")
            
            merge_threshold = st.slider("相似度阈值 (Threshold)", 0.80, 1.00, 0.92, 0.01, help="高于此相似度的簇将被合并。推荐 0.90 - 0.95")
            
            if st.button("🧩 开始合并相似簇", type="primary"):
                with st.spinner("正在计算簇中心并合并..."):
                    # 调用合并函数
                    engine.merge_similar_clusters(threshold=merge_threshold)
                    
                    if enable_auto_save:
                        engine.save_state(STATE_FILE)
                        st.success("✅ 合并操作完成 (状态已自动保存)")
                    else:
                        st.success("✅ 合并操作完成 (仅内存更新)")
                        
                    time.sleep(1)
                    st.rerun()

        # === Tab 3: 结果预览 ===
        with tab3:
            results_df = engine.get_results()
            if results_df is not None:
                # 统计
                n_clusters = len(results_df[results_df['Cluster'] != -1]['Cluster'].unique())
                n_noise = (results_df['Cluster'] == -1).sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("总行数", len(results_df))
                c2.metric("当前簇数量", n_clusters)
                c3.metric("噪音数据数", n_noise)
                
                st.dataframe(results_df.head(100), use_container_width=True)
            else:
                st.warning("暂无结果数据")

        # === Tab 4: 导出 ===
        with tab4:
            results_df = engine.get_results()
            if results_df is not None:
                fname = st.text_input("导出Excel文件名", "final_cluster_result.xlsx")
                if st.button("💾 保存Excel到磁盘"):
                    save_path = os.path.join(os.path.dirname(OUTPUT_DIR), fname)
                    engine.save_results(save_path)
                    st.success(f"已保存至: {save_path}")
                
                # CSV下载
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 下载CSV文件", data=csv_data, file_name="cluster_result.csv", mime="text/csv")

    else:
        # 如果没有加载数据，显示占位提示
        if input_file is None:
            st.info("👈 请在上方选择 [加载历史状态] 或 [上传新文件] 以启用控制台。")

# =====================================================================
# 聚类评估页面
# =====================================================================
elif page == "📊 聚类评估":
    st.title("📊 聚类效果评估")
    st.markdown("---")
    
    # [新增] 优先检查 Session State
    if 'cluster_engine' in st.session_state and st.session_state['cluster_engine'] is not None:
        st.info("💡 正在使用内存中的聚类分析结果。")
        engine = st.session_state['cluster_engine']
        results_df = engine.get_results()
        embeddings = engine.get_embeddings()
        
        # 直接开始评估，无需上传
        evaluator = ClusterEvaluator(results_df, embeddings)
        metrics = evaluator.compute_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("总样本数", metrics['Total Samples'])
        with col2: st.metric("聚类个数", metrics['Valid Clusters'])
        with col3: st.metric("噪音比例", metrics['Noise Ratio'])
        
        if 'Silhouette Score' in metrics and metrics['Silhouette Score'] != "N/A (簇数量不足)":
            col1, col2 = st.columns(2)
            with col1: st.metric("轮廓系数", metrics['Silhouette Score'])
            with col2: st.metric("CH分数", metrics['CH Score'])
            
        # 相似度分析 (如果有)
        if st.button("计算相似度矩阵 (可能较慢)"):
            with st.spinner("计算中..."):
                evaluator.analyze_similarity()
    else:
        # 回退到上传文件模式
        input_file = st.file_uploader("上传聚类结果文件 (必须包含 Text 和 Cluster 列)", type=["xlsx", "xls"], key="eval_input")
        
        if input_file is not None:
            try:
                with st.spinner("📖 读取文件..."):
                    results_df = pd.read_excel(input_file)
                
                # 检查必需列
                if 'Text' not in results_df.columns or 'Cluster' not in results_df.columns:
                    st.error("❌ 文件必须包含 'Text' 和 'Cluster' 列")
                else:
                    with st.spinner("🤖 加载SBERT模型并计算向量..."):
                        engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
                        texts = results_df['Text'].tolist()
                        embeddings = engine.model.encode(texts, show_progress_bar=False)
                    
                    evaluator = ClusterEvaluator(results_df, embeddings)
                    # ... (后续显示逻辑同上)
                    metrics = evaluator.compute_metrics()
                    st.write(metrics)
                    
            except Exception as e:
                st.error(f"❌ 评估失败: {str(e)}")

# =====================================================================
# LLM关键词提取页面
# =====================================================================
elif page == "🏷️ LLM关键词提取":
    st.title("🏷️ LLM关键词提取")
    st.markdown("使用大模型智能归纳聚类关键词")
    st.markdown("---")
    
    st.subheader("🔑 API配置")
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("API Key", type="password", help="输入你的DeepSeek/OpenAI API Key")
    with col2:
        base_url = st.text_input("API地址", "https://api.deepseek.com", help="API端点地址")
    
    st.subheader("📁 数据源")
    input_file = st.file_uploader("上传聚类结果", type=["xlsx", "xls"], key="llm_input")
    
    if input_file is not None and api_key and base_url:
        if st.button("🚀 开始提取关键词"):
            try:
                with st.spinner("📖 读取文件..."):
                    df = pd.read_excel(input_file)
                
                with st.spinner("🤖 初始化LLM提取器..."):
                    extractor = LLMKeywordExtractor(
                        api_key=api_key,
                        base_url=base_url,
                        model="deepseek-chat"
                    )
                
                with st.spinner("⏳ 正在调用LLM提取关键词... (请耐心等待)"):
                    df_result = extractor.extract_keywords(df, text_col='Text')
                
                # 保存结果
                output_path = os.path.join(OUTPUT_DIR, "llm_keywords_result.xlsx")
                save_excel(df_result, output_path)
                
                # 显示结果
                st.subheader("📋 提取结果")
                display_df = df_result[['Cluster', 'LLM_Keywords']].drop_duplicates()
                st.dataframe(display_df, use_container_width=True)
                
                st.success(f"✅ 结果已保存到: {output_path}")
                
            except Exception as e:
                st.error(f"❌ 提取失败: {str(e)}")
    else:
        st.warning("⚠️ 请填写API配置和上传文件")

# =====================================================================
# 结果查看页面
# =====================================================================
elif page == "📈 结果查看":
    st.title("📈 结果查看与导出")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 加载文件")
        result_file = st.file_uploader("选择结果文件", type=["xlsx", "xls"])
    
    with col2:
        st.subheader("🔍 过滤选项")
        filter_noise = st.checkbox("隐藏噪音数据", value=False)
    
    if result_file is not None:
        try:
            df = pd.read_excel(result_file)
            
            if filter_noise and 'Cluster' in df.columns:
                df = df[df['Cluster'] != -1]
            
            st.subheader("📊 统计信息")
            st.metric("总行数", len(df))
            
            # 显示数据
            st.dataframe(df, use_container_width=True, height=400)
            
            # 导出
            st.subheader("💾 导出选项")
            export_filename = st.text_input("导出文件名", "result_export.xlsx")
            if st.button("💾 导出为Excel"):
                export_path = os.path.join(OUTPUT_DIR, export_filename)
                save_excel(df, export_path)
                st.success(f"✅ 已导出到: {export_path}")
        
        except Exception as e:
            st.error(f"❌ 加载失败: {str(e)}")

# =====================================================================
# 页脚
# =====================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
    <p>🎯 税务诉求分析系统 Pro v2.0</p>
    </div>
""", unsafe_allow_html=True)