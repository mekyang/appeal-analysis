from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import os
import io

# 导入你的其他模块
from config import *
from extract_content import ContentExtractor_12366, ContentExtractor_12345, ContentExtractor_ZN
from data_sanitizer import TaxDataSanitizer
from data_analysis import TaxClusteringEngine
from cluster_eval import ClusterEvaluator
from cluster_tagger import LLMKeywordExtractor
from excel_handle import *

app = FastAPI(
    title="税务诉求分析系统",
    description="基于FastAPI和Vue的税务诉求分析系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "欢迎使用税务诉求分析系统"}

# =====================================================================
# 数据预处理接口
# =====================================================================
@app.post("/api/preprocess")
async def preprocess_file(
    file: UploadFile = File(...),
    extractor_type: str = Form(..., description="提取器类型: 12366, 12345, zn"),
    use_ner: bool = Form(True, description="是否启用NER模型脱敏"),
    column_name: str = Form("业务内容", description="内容列名")
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="只支持Excel文件")
    
    # 保存上传的文件
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # 读取Excel文件
        df = read_excel(file_path, sheet_name=0, header=0)
        df['Trace_ID'] = df.index + 2
        
        # 初始化工具链
        if extractor_type == "12366":
            extractor = ContentExtractor_12366()
        elif extractor_type == "12345":
            extractor = ContentExtractor_12345()
        elif extractor_type == "zn":
            extractor = ContentExtractor_ZN()
        else:
            raise HTTPException(status_code=400, detail="无效的提取器类型")
        
        sanitizer = TaxDataSanitizer(use_ner=use_ner)
        
        # 提取内容
        raw_content_series = extractor.extract_content(df, column_name)
        
        # 隐私脱敏
        temp_df = pd.DataFrame({'temp_text': raw_content_series})
        sanitizer.process_dataframe(temp_df, 'temp_text', 'Sanitized_Content')
        
        # 构建输出DataFrame
        output_df = pd.DataFrame({
            'Trace_ID': df['Trace_ID'],
            'Original_Content': df[column_name],
            'Extracted_Content': raw_content_series,
            'Sanitized_Content': temp_df['Sanitized_Content']
        })
        
        # 过滤无效空行
        initial_count = len(output_df)
        output_df = output_df[output_df['Sanitized_Content'].str.strip() != '']
        final_count = len(output_df)
        
        # 保存处理结果
        output_path = os.path.join("temp", f"processed_{file.filename}")
        save_excel(output_df, output_path)
        
        # 准备下载
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_buffer.seek(0)
        
        return {
            "message": "文件处理成功",
            "total_rows": initial_count,
            "valid_rows": final_count,
            "removed_rows": initial_count - final_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

# =====================================================================
# 文本聚类分析接口
# =====================================================================
@app.post("/api/cluster")
async def cluster_data(
    file: UploadFile = File(...),
    text_column: str = Form("Sanitized_Content", description="文本列名"),
    original_column: str = Form("业务编号", description="ID列名"),
    n_neighbors: int = Form(15, description="n_neighbors参数"),
    n_components: int = Form(5, description="n_components参数"),
    min_cluster_size: int = Form(10, description="min_cluster_size参数"),
    keyword_top_n: int = Form(5, description="关键词数"),
    auto_save: bool = Form(True, description="是否自动保存状态")
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="只支持Excel文件")
    
    # 保存上传的文件
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # 读取Excel文件
        df = read_excel(file_path, sheet_name=0, header=0)
        
        # 检查文本列是否存在
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"文件中不存在列: {text_column}")
        
        # 准备文本数据
        texts = df[text_column].dropna().astype(str).tolist()
        texts = [t for t in texts if len(t.strip()) > 0]
        
        # 尝试获取原始ID列
        original_column_list = None
        for col in ['Trace_ID', '工单编号', 'ID', '序号', original_column]:
            if col in df.columns:
                original_column_list = df[col].tolist()
                break
        
        # 初始化聚类引擎
        engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
        
        # 运行分析
        engine.run_analysis(
            texts,
            original=original_column_list,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_cluster_size=min_cluster_size,
            keyword_top_n=keyword_top_n
        )
        
        # 保存状态
        if auto_save:
            engine.save_state(STATE_FILE)
        
        # 获取结果
        results_df = engine.get_results()
        
        # 统计信息
        n_clusters = len(results_df[results_df['Cluster'] != -1]['Cluster'].unique())
        n_noise = int((results_df['Cluster'] == -1).sum())
        
        # 保存结果
        output_path = os.path.join("temp", f"clustered_{file.filename}")
        save_excel(results_df, output_path)
        
        return {
            "message": "聚类分析成功",
            "total_texts": len(texts),
            "n_clusters": int(n_clusters),
            "n_noise": n_noise,
            "output_file": output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

# =====================================================================
# 聚类评估接口
# =====================================================================
@app.post("/api/evaluate")
async def evaluate_cluster(
    file: UploadFile = File(...),
    text_column: str = Form("Text", description="文本列名"),
    cluster_column: str = Form("Cluster", description="聚类列名")
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="只支持Excel文件")
    
    # 保存上传的文件
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # 读取Excel文件
        results_df = read_excel(file_path, sheet_name=0, header=0)
        
        # 检查必需列
        if text_column not in results_df.columns:
            raise HTTPException(status_code=400, detail=f"文件中不存在列: {text_column}")
        if cluster_column not in results_df.columns:
            raise HTTPException(status_code=400, detail=f"文件中不存在列: {cluster_column}")
        
        # 初始化聚类引擎获取嵌入
        engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
        texts = results_df[text_column].tolist()
        embeddings = engine.model.encode(texts, show_progress_bar=False)
        
        # 初始化评估器
        evaluator = ClusterEvaluator(results_df, embeddings)
        metrics = evaluator.compute_metrics()
        
        return {
            "message": "评估成功",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

# =====================================================================
# LLM关键词提取接口
# =====================================================================
@app.post("/api/extract-keywords")
async def extract_keywords(
    file: UploadFile = File(...),
    api_key: str = Form(..., description="API Key"),
    base_url: str = Form("https://api.deepseek.com", description="API地址"),
    text_col: str = Form("Text", description="文本列名")
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="只支持Excel文件")
    
    # 保存上传的文件
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # 读取Excel文件
        df = read_excel(file_path, sheet_name=0, header=0)
        
        # 检查文本列是否存在
        if text_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"文件中不存在列: {text_col}")
        
        # 初始化LLM提取器
        extractor = LLMKeywordExtractor(
            api_key=api_key,
            base_url=base_url,
            model="deepseek-chat"
        )
        
        # 提取关键词
        df_result = extractor.extract_keywords(df, text_col=text_col)
        
        # 保存结果
        output_path = os.path.join("temp", f"keywords_{file.filename}")
        save_excel(df_result, output_path)
        
        # 准备返回数据
        display_df = df_result[['Cluster', 'LLM_Keywords']].drop_duplicates()
        keywords_result = display_df.to_dict('records')
        
        return {
            "message": "关键词提取成功",
            "result": keywords_result,
            "output_file": output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关键词提取失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

# =====================================================================
# 结果下载接口
# =====================================================================
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("temp", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 读取文件
    with open(file_path, "rb") as f:
        content = f.read()
    
    # 创建响应
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# =====================================================================
# 加载历史状态接口
# =====================================================================
@app.get("/api/load-state")
async def load_state():
    if not os.path.exists(STATE_FILE):
        raise HTTPException(status_code=404, detail="状态文件不存在")
    
    try:
        engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
        if engine.load_state(STATE_FILE):
            results_df = engine.get_results()
            n_clusters = len(results_df[results_df['Cluster'] != -1]['Cluster'].unique())
            n_noise = (results_df['Cluster'] == -1).sum()
            
            return {
                "message": "状态加载成功",
                "total_rows": len(results_df),
                "n_clusters": n_clusters,
                "n_noise": n_noise
            }
        else:
            raise HTTPException(status_code=500, detail="状态加载失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载状态失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
