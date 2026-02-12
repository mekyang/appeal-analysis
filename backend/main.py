import uuid
import os
import io
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 导入自定义模块
from config import *
from extract_content import ContentExtractor_12366, ContentExtractor_12345, ContentExtractor_ZN
from data_sanitizer import TaxDataSanitizer
from data_analysis import TaxClusteringEngine
from cluster_eval import ClusterEvaluator
from cluster_tagger import LLMKeywordExtractor
from excel_handle import *

app = FastAPI(title="税务诉求分析系统", description="异步修复纯净版", version="3.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 全局变量 ---
TASKS: Dict[str, Dict[str, Any]] = {}
GLOBAL_SANITIZER = None

def get_sanitizer(use_ner: bool):
    """全局复用 NER 模型，加速 Step 1"""
    global GLOBAL_SANITIZER
    if GLOBAL_SANITIZER is None:
        print("🚀 [System] 初始化全局 NER 模型...")
        GLOBAL_SANITIZER = TaxDataSanitizer(use_ner=use_ner)
    return GLOBAL_SANITIZER

@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"progress": 0, "status": "not_found"}
    return task

@app.get("/")
async def root():
    return {"message": "欢迎使用税务诉求分析系统"}

# =====================================================================
# 1. 后台任务：数据预处理
# =====================================================================
def background_preprocess_task(task_id: str, file_path: str, params: dict):
    try:
        def progress_callback(curr, total, msg):
            if total > 0:
                p = int(20 + (curr / total) * 70)
                TASKS[task_id]["progress"] = p
                TASKS[task_id]["msg"] = msg

        TASKS[task_id].update({"progress": 5, "msg": "读取文件中..."})
        df = read_excel(file_path, sheet_name=0, header=0)
        df['Trace_ID'] = df.index + 2
        
        TASKS[task_id].update({"progress": 10, "msg": "提取文本..."})
        extractor_type = params['extractor_type']
        column_name = params['column_name']
        
        if extractor_type == "12366": extractor = ContentExtractor_12366()
        elif extractor_type == "12345": extractor = ContentExtractor_12345()
        elif extractor_type == "zn": extractor = ContentExtractor_ZN()
        else: raise Exception("无效提取器")
        
        raw_content_series = extractor.extract_content(df, column_name)
        
        TASKS[task_id].update({"progress": 20, "msg": "加载模型脱敏中..."})
        sanitizer = get_sanitizer(use_ner=params['use_ner'])
        
        temp_df = pd.DataFrame({'temp_text': raw_content_series})
        sanitizer.process_dataframe(temp_df, 'temp_text', 'Sanitized_Content', progress_callback=progress_callback)
        
        TASKS[task_id].update({"progress": 95, "msg": "保存结果..."})
        output_df = pd.DataFrame({
            'Trace_ID': df['Trace_ID'],
            'Original_Content': df[column_name],
            'Extracted_Content': raw_content_series,
            'Sanitized_Content': temp_df['Sanitized_Content']
        })
        
        initial_count = len(output_df)
        output_df = output_df[output_df['Sanitized_Content'].str.strip() != '']
        final_count = len(output_df)
        
        output_filename = f"processed_{os.path.basename(file_path).split('_', 1)[-1]}"
        output_path = os.path.join("temp", output_filename)
        save_excel(output_df, output_path)
        
        TASKS[task_id].update({
            "status": "done", "progress": 100, "msg": "预处理完成",
            "result": {
                "message": "处理成功", "total_rows": initial_count,
                "valid_rows": final_count, "removed_rows": initial_count - final_count,
                "output_file": output_path
            }
        })
    except Exception as e:
        print(f"Preprocess Error: {e}")
        TASKS[task_id].update({"status": "error", "error": str(e)})
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.post("/api/preprocess")
async def preprocess_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extractor_type: str = Form(...),
    use_ner: bool = Form(True),
    column_name: str = Form("业务内容")
):
    os.makedirs("temp", exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join("temp", f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f: f.write(await file.read())

    task_id = file_id
    TASKS[task_id] = {"status": "running", "progress": 0, "msg": "准备开始..."}
    background_tasks.add_task(background_preprocess_task, task_id, file_path, {
        "extractor_type": extractor_type, "use_ner": use_ner, "column_name": column_name
    })
    return {"task_id": task_id, "message": "任务已启动"}

# =====================================================================
# 2. 后台任务：聚类分析 (已移除所有副作用代码)
# =====================================================================
def background_cluster_task(task_id: str, file_path: str, params: dict):
    try:
        def progress_callback(curr, total, msg):
            TASKS[task_id]["progress"] = int(curr)
            TASKS[task_id]["msg"] = msg
        
        df = read_excel(file_path, sheet_name=0, header=0)
        
        text_col = params.get('text_column', 'Sanitized_Content')
        if text_col not in df.columns:
            raise Exception(f"找不到列名: {text_col}")

        # --- 简单且稳健的数据对齐 ---
        # 你的原始代码这里可能会因为空行导致不对齐，这里加一个保险的清洗逻辑
        df_clean = df.dropna(subset=[text_col])
        df_clean = df_clean[df_clean[text_col].astype(str).str.strip().str.len() > 0]
        
        if len(df_clean) == 0:
            raise Exception("没有有效的文本数据可用于聚类")

        texts = df_clean[text_col].astype(str).tolist()
        
        # 尝试获取原始ID (逻辑与你原始代码意图一致，但修复了潜在对齐问题)
        original_col_name = params.get('original_column')
        original_list = None
        candidates = [original_col_name, 'Trace_ID', '业务编号', 'ID', '序号']
        for col in candidates:
            if col and col in df_clean.columns:
                original_list = df_clean[col].tolist()
                break

        # --- 关键：完全信任参数，不修改 ---
        n_neighbors = params['n_neighbors']
        min_cluster_size = params['min_cluster_size']
        
        print(f"⚡ 开始聚类: 数据={len(texts)}条, Neighbors={n_neighbors}, MinSize={min_cluster_size}")

        engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
        
        engine.run_analysis(
            texts,
            original=original_list,
            n_neighbors=n_neighbors,
            n_components=params['n_components'],
            min_cluster_size=min_cluster_size,
            keyword_top_n=params['keyword_top_n'],
            progress_callback=progress_callback
        )
        
        output_filename = f"clustered_{os.path.basename(file_path).split('_', 1)[-1]}"
        output_path = os.path.join("temp", output_filename)
        save_excel(engine.get_results(), output_path)
        
        results_df = engine.get_results()
        n_clusters = len(results_df[results_df['Cluster'] != -1]['Cluster'].unique())
        n_noise = int((results_df['Cluster'] == -1).sum())

        TASKS[task_id].update({
            "status": "done", "progress": 100, "msg": "聚类完成",
            "result": {
                "message": "聚类分析成功",
                "total_texts": len(texts),
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "output_file": output_path
            }
        })
    except Exception as e:
        print(f"Cluster Error: {e}")
        TASKS[task_id].update({"status": "error", "error": str(e)})
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.post("/api/cluster")
async def cluster_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_column: str = Form("Sanitized_Content"),
    original_column: str = Form("业务编号"),
    n_neighbors: int = Form(15),
    n_components: int = Form(5),
    min_cluster_size: int = Form(10),
    keyword_top_n: int = Form(5),
    auto_save: bool = Form(True)
):
    os.makedirs("temp", exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join("temp", f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f: f.write(await file.read())
    
    task_id = file_id
    TASKS[task_id] = {"status": "running", "progress": 0, "msg": "准备开始..."}
    
    params = {
        "text_column": text_column, "original_column": original_column,
        "n_neighbors": n_neighbors, "n_components": n_components,
        "min_cluster_size": min_cluster_size, "keyword_top_n": keyword_top_n
    }
    background_tasks.add_task(background_cluster_task, task_id, file_path, params)
    return {"task_id": task_id, "message": "任务已启动"}

# =====================================================================
# 3. 后台任务：LLM 摘要
# =====================================================================
def background_keywords_task(task_id: str, file_path: str, params: dict):
    try:
        def progress_callback(curr, total, msg):
            if total > 0:
                p = int((curr / total) * 100)
                TASKS[task_id]["progress"] = p
                TASKS[task_id]["msg"] = msg

        df = read_excel(file_path, sheet_name=0, header=0)
        TASKS[task_id].update({"progress": 5, "msg": "连接 LLM..."})
        
        extractor = LLMKeywordExtractor(
            api_key=params['api_key'], base_url=params['base_url'], model="deepseek-chat"
        )
        
        df_result = extractor.extract_keywords(
            df, text_col=params['text_col'], progress_callback=progress_callback
        )
        
        output_filename = f"keywords_{os.path.basename(file_path).split('_', 1)[-1]}"
        output_path = os.path.join("temp", output_filename)
        save_excel(df_result, output_path)
        
        display_df = df_result[['Cluster', 'LLM_Keywords']].drop_duplicates()
        if 'Count' not in display_df.columns:
            count_series = df_result['Cluster'].value_counts()
            display_df['Count'] = display_df['Cluster'].map(count_series)

        TASKS[task_id].update({
            "status": "done", "progress": 100, "msg": "摘要完成",
            "result": {
                "message": "成功", "result": display_df.to_dict('records'),
                "output_file": output_path
            }
        })
    except Exception as e:
        print(f"LLM Error: {e}")
        TASKS[task_id].update({"status": "error", "error": str(e)})
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.post("/api/extract-keywords")
async def extract_keywords(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    base_url: str = Form(...),
    text_col: str = Form("Text")
):
    os.makedirs("temp", exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join("temp", f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f: f.write(await file.read())

    task_id = file_id
    TASKS[task_id] = {"status": "running", "progress": 0, "msg": "准备开始..."}
    background_tasks.add_task(background_keywords_task, task_id, file_path, {
        "api_key": api_key, "base_url": base_url, "text_col": text_col
    })
    return {"task_id": task_id}

# --- 辅助接口 ---
@app.post("/api/evaluate")
async def evaluate_cluster(
    file: UploadFile = File(...), text_column: str = Form("Text"), cluster_column: str = Form("Cluster")
):
    if not file.filename.endswith((".xlsx", ".xls")): raise HTTPException(status_code=400)
    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f: f.write(await file.read())
    try:
        results_df = read_excel(file_path, sheet_name=0, header=0)
        engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)
        texts = results_df[text_column].tolist()
        embeddings = engine.model.encode(texts, show_progress_bar=False)
        evaluator = ClusterEvaluator(results_df, embeddings)
        return {"message": "评估成功", "metrics": evaluator.compute_metrics()}
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("temp", filename)
    if not os.path.exists(file_path): raise HTTPException(status_code=404)
    with open(file_path, "rb") as f: content = f.read()
    return StreamingResponse(io.BytesIO(content), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={filename}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)