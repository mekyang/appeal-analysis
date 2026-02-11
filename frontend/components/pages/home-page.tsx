"use client";

import { useState, useMemo } from "react";
import type { PageKey } from "@/components/app-sidebar";
import { PipelineStepCard, type StepStatus } from "@/components/pipeline-step-card";
import { ConfigDialog } from "@/components/config-dialog";
import { FileUpload } from "@/components/file-upload";
import { MetricCard } from "@/components/metric-card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
// import { Progress } from "@/components/ui/progress"; //这行暂时不需要，用原生div模拟了
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  preprocessFile,
  clusterData,
  extractKeywords,
  downloadFile,
  type PreprocessResponse,
  type ClusterResponse,
  type KeywordsResponse,
} from "@/lib/api";
import {
  ChevronRight,
  FileSpreadsheet,
  Layers,
  Hash,
  Volume2,
  Download,
  Trash2,
  Rows3,
  Brain,
  Tags,
  BarChart3,
  AlertCircle,
  Target,
} from "lucide-react";

// --- 引入图表库 ---
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

// ===================================================================
// Types
// ===================================================================

interface CleanConfig {
  extractor: string;
  useNer: boolean;
  column: string;
}

interface ClusterConfig {
  nNeighbors: number;
  nComponents: number;
  minCluster: number;
  topN: number;
  textColumn: string;
  originalColumn: string;
}

interface LlmConfig {
  apiKey: string;
  baseUrl: string;
}

interface HomePageProps {
  onNavigate: (page: PageKey) => void;
}

// ===================================================================
// Component
// ===================================================================

export function HomePage({ onNavigate }: HomePageProps) {
  // --- Pipeline state ---
  const [file, setFile] = useState<File | null>(null);
  const [preprocessResult, setPreprocessResult] = useState<PreprocessResponse | null>(null);
  const [clusterResult, setClusterResult] = useState<ClusterResponse | null>(null);
  const [keywordsResult, setKeywordsResult] = useState<KeywordsResponse | null>(null);

  // --- Filename State ---
  const [realClusteredFilename, setRealClusteredFilename] = useState<string | null>(null);
  const [realKeywordsFilename, setRealKeywordsFilename] = useState<string | null>(null);

  // --- Loading / progress ---
  const [step1Loading, setStep1Loading] = useState(false);
  const [step1Progress, setStep1Progress] = useState(0);
  const [step1ProgressText, setStep1ProgressText] = useState("");
  const [step2Loading, setStep2Loading] = useState(false);
  const [step2Progress, setStep2Progress] = useState(0);
  const [step2ProgressText, setStep2ProgressText] = useState("");
  const [step3Loading, setStep3Loading] = useState(false);
  const [step3Progress, setStep3Progress] = useState(0);
  const [step3ProgressText, setStep3ProgressText] = useState("");

  // --- Error ---
  const [error, setError] = useState<string | null>(null);

  // --- Config dialogs ---
  const [cleanDialogOpen, setCleanDialogOpen] = useState(false);
  const [clusterDialogOpen, setClusterDialogOpen] = useState(false);
  const [llmDialogOpen, setLlmDialogOpen] = useState(false);

  // --- Config values ---
  const [cleanCfg, setCleanCfg] = useState<CleanConfig>({
    extractor: "12366",
    useNer: true,
    column: "业务内容",
  });
  const [clusterCfg, setClusterCfg] = useState<ClusterConfig>({
    nNeighbors: 15,
    nComponents: 5,
    minCluster: 10,
    topN: 5,
    textColumn: "Sanitized_Content",
    originalColumn: "业务编号",
  });
  const [llmCfg, setLlmCfg] = useState<LlmConfig>({
    apiKey: "sk-f1fec6b90628475ba7ce12b2c389c85a",
    baseUrl: "https://api.deepseek.com",
  });

  // Temp config
  const [tempClean, setTempClean] = useState<CleanConfig>(cleanCfg);
  const [tempCluster, setTempCluster] = useState<ClusterConfig>(clusterCfg);
  const [tempLlm, setTempLlm] = useState<LlmConfig>(llmCfg);

  // --- Status ---
  const step1Status: StepStatus = step1Loading ? "running" : preprocessResult ? "done" : file ? "ready" : "ready";
  const step2Status: StepStatus = step2Loading ? "running" : clusterResult ? "done" : preprocessResult ? "ready" : "waiting";
  const step3Status: StepStatus = step3Loading ? "running" : keywordsResult ? "done" : clusterResult ? "ready" : "waiting";

  // --- Pipeline file tracking (Fallback) ---
  const processedFilename = file ? `processed_${file.name}` : null;
  const fallbackClusteredFilename = file ? `clustered_${file.name}` : null;
  const fallbackKeywordsFilename = file ? `keywords_${file.name}` : null;

  // --- Step 1: Preprocess ---
  const handleStep1 = async () => {
    if (!file) return;
    setStep1Loading(true);
    setError(null);
    setPreprocessResult(null);
    setClusterResult(null);
    setKeywordsResult(null);
    setRealClusteredFilename(null);
    setRealKeywordsFilename(null);

    try {
      setStep1Progress(15);
      setStep1ProgressText("正在上传文件...");
      await new Promise((r) => setTimeout(r, 300));
      setStep1Progress(40);
      setStep1ProgressText(`正在使用提取器处理数据...`);

      const res = await preprocessFile(file, cleanCfg.extractor, cleanCfg.useNer, cleanCfg.column);

      setStep1Progress(100);
      setStep1ProgressText("处理完成");
      setPreprocessResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "数据处理失败");
    } finally {
      setStep1Loading(false);
    }
  };

  // --- Step 2: Cluster ---
  const handleStep2 = async () => {
    if (!processedFilename) return;
    setStep2Loading(true);
    setError(null);
    setClusterResult(null);
    setKeywordsResult(null);
    setRealKeywordsFilename(null);

    try {
      setStep2Progress(10);
      setStep2ProgressText("正在下载预处理文件...");

      const blob = await downloadFile(processedFilename);
      const processedFile = new File([blob], processedFilename, {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });

      setStep2Progress(30);
      setStep2ProgressText("SBERT 编码 + HDBSCAN 聚类中...");

      const res = await clusterData(
        processedFile,
        clusterCfg.textColumn,
        clusterCfg.originalColumn,
        clusterCfg.nNeighbors,
        clusterCfg.nComponents,
        clusterCfg.minCluster,
        clusterCfg.topN,
        true
      );

      setStep2Progress(100);
      setStep2ProgressText("聚类完成");
      
      if (res && (res as any).output_file) {
        const fullPath = (res as any).output_file as string;
        const realName = fullPath.split(/[/\\]/).pop();
        if (realName) {
            setRealClusteredFilename(realName);
        }
      }
      
      setClusterResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "聚类分析失败");
    } finally {
      setStep2Loading(false);
    }
  };

  // --- Step 3: LLM Keywords ---
  const handleStep3 = async () => {
    const targetFilename = realClusteredFilename || fallbackClusteredFilename;
    if (!targetFilename) {
        setError("无法找到聚类文件，请重新执行第二步");
        return;
    }

    if (!llmCfg.apiKey) {
      setLlmDialogOpen(true);
      return;
    }
    setStep3Loading(true);
    setError(null);
    setKeywordsResult(null);

    try {
      setStep3Progress(15);
      setStep3ProgressText("正在下载聚类文件...");

      const blob = await downloadFile(targetFilename);
      const clusterFile = new File([blob], targetFilename, {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });

      setStep3Progress(40);
      setStep3ProgressText("AI 正在阅读并生成标签 (请稍候)...");

      const res = await extractKeywords(clusterFile, llmCfg.apiKey, llmCfg.baseUrl, "Text");

      setStep3Progress(100);
      setStep3ProgressText("摘要生成完成");
      
      if (res && (res as any).output_file) {
        const fullPath = (res as any).output_file as string;
        const realName = fullPath.split(/[/\\]/).pop();
        if (realName) {
            setRealKeywordsFilename(realName);
        }
      }

      setKeywordsResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "LLM 调用失败");
    } finally {
      setStep3Loading(false);
    }
  };

  const handleDownloadResult = async (filename: string | null) => {
    if (!filename) return;
    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    window.open(`${API_BASE}/api/download/${filename}`, "_blank");
  };

  // ===================================================================
  // 📊 Dashboard Logic (数据分析引擎) - 【这里是修改核心】
  // ===================================================================
  const reportData = useMemo(() => {
    if (!keywordsResult || !keywordsResult.result) return null;

    const rawData = keywordsResult.result;

    // --- ❌ 删除旧逻辑 (手动累加导致全为1) ---
    // const clusterMap = new Map... rawData.forEach...

    // --- ✅ 新逻辑: 直接读取后端返回的 Count 字段 ---
    const analysisList = rawData.map((item: any) => ({
        clusterId: item.Cluster,
        keyword: item.LLM_Keywords,
        // 核心修正：优先读取 Count。
        // 如果后端没返回 Count (可能是旧接口)，则回退到 1，但正常情况这里会读到 50, 100 等真实数值
        count: item.Count ? Number(item.Count) : 1
    }));

    // 2. 计算总记录数
    const totalRecords = analysisList.reduce((sum, item) => sum + item.count, 0);

    // 3. 计算占比
    const processedList = analysisList.map(item => ({
        ...item,
        percentage: totalRecords > 0 ? item.count / totalRecords : 0
    }));

    const validClusters = processedList.filter(i => i.clusterId !== -1);
    const noiseCluster = processedList.find(i => i.clusterId === -1);

    // 4. 排序 (Sort by Count Desc)
    validClusters.sort((a, b) => b.count - a.count);

    // 5. Metrics
    const validCount = validClusters.reduce((sum, item) => sum + item.count, 0);
    const noiseCount = noiseCluster ? noiseCluster.count : 0;
    const coverage = totalRecords > 0 ? validCount / totalRecords : 0;
    const topicCount = validClusters.length;

    // 6. Top 10 Data
    const top10 = validClusters.slice(0, 10).map(item => ({
        name: item.keyword.length > 8 ? item.keyword.substring(0, 8) + '...' : item.keyword,
        fullName: item.keyword,
        count: item.count,
        fill: "hsl(var(--primary))"
    }));

    // 7. Full List
    const fullDisplayList = [...validClusters];
    if (noiseCluster) {
        fullDisplayList.push({
            ...noiseCluster,
            keyword: "⚠️ 离散/无法归类的数据 (噪音)"
        });
    }

    return {
        totalRecords,
        topicCount,
        coverage,
        noiseCount,
        top10,
        fullDisplayList
    };

  }, [keywordsResult]);

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold text-foreground text-balance">
          税费诉求分析模型
        </h1>
        <p className="text-sm text-muted-foreground leading-relaxed">
          上传数据，一键流转：
          <span className="font-medium text-foreground">清洗</span>
          {" -> "}
          <span className="font-medium text-foreground">聚类</span>
          {" -> "}
          <span className="font-medium text-foreground">摘要</span>
        </p>
      </div>

      {/* Pipeline Steps (Clean / Cluster / Summary) */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr_auto_1fr] gap-4 items-start">
        {/* Step 1 */}
        <PipelineStepCard
          stepNumber={1}
          title="诉求数据清洗"
          status={step1Status}
          statusText={preprocessResult ? `已处理: ${preprocessResult.valid_rows} 条` : undefined}
          onConfigure={() => { setTempClean(cleanCfg); setCleanDialogOpen(true); }}
          onRun={handleStep1}
          runLabel="开始处理"
          runDisabled={!file}
          progress={step1Loading ? step1Progress : undefined}
          progressText={step1Loading ? step1ProgressText : undefined}
        >
          <FileUpload file={file} onFileChange={setFile} compact />
        </PipelineStepCard>
        
        <div className="hidden lg:flex items-center justify-center pt-16">
          <ChevronRight className={`h-6 w-6 ${preprocessResult ? "text-[hsl(var(--success))] animate-pulse-dot" : "text-border"}`} />
        </div>

        {/* Step 2 */}
        <PipelineStepCard
          stepNumber={2}
          title="诉求数据自动聚类"
          status={step2Status}
          statusText={clusterResult ? `发现 ${clusterResult.n_clusters} 个类别` : undefined}
          onConfigure={() => { setTempCluster(clusterCfg); setClusterDialogOpen(true); }}
          onRun={handleStep2}
          runLabel="执行聚类"
          progress={step2Loading ? step2Progress : undefined}
          progressText={step2Loading ? step2ProgressText : undefined}
        />

        <div className="hidden lg:flex items-center justify-center pt-16">
          <ChevronRight className={`h-6 w-6 ${clusterResult ? "text-[hsl(var(--success))] animate-pulse-dot" : "text-border"}`} />
        </div>

        {/* Step 3 */}
        <PipelineStepCard
          stepNumber={3}
          title="诉求数据智能摘要"
          status={step3Status}
          statusText={keywordsResult ? "摘要生成完毕" : undefined}
          onConfigure={() => { setTempLlm(llmCfg); setLlmDialogOpen(true); }}
          onRun={handleStep3}
          runLabel="生成摘要"
          progress={step3Loading ? step3Progress : undefined}
          progressText={step3Loading ? step3ProgressText : undefined}
        />
      </div>

      {/* Error Display */}
      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-destructive">处理失败</p>
            <p className="text-sm text-destructive/80 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* ============================================================= */}
      {/* 📊 Real-time Dashboard (复刻 Streamlit 看板)                   */}
      {/* ============================================================= */}
      <div className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-foreground">实时数据看板</h2>

        <Tabs defaultValue="tab-clean" className="w-full">
          <TabsList className="w-full justify-start bg-muted/50 p-1 rounded-lg">
            <TabsTrigger value="tab-clean" className="gap-1.5 text-xs">
              <FileSpreadsheet className="h-3.5 w-3.5" />
              1. 清洗结果
            </TabsTrigger>
            <TabsTrigger value="tab-cluster" className="gap-1.5 text-xs">
              <Brain className="h-3.5 w-3.5" />
              2. 聚类分布
            </TabsTrigger>
            <TabsTrigger value="tab-report" className="gap-1.5 text-xs">
              <BarChart3 className="h-3.5 w-3.5" />
              3. 分析报告
            </TabsTrigger>
          </TabsList>

          {/* Tab 1: Clean Results */}
          <TabsContent value="tab-clean" className="mt-4">
            {preprocessResult ? (
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <MetricCard
                    label="原始行数"
                    value={preprocessResult.total_rows.toLocaleString()}
                    icon={<FileSpreadsheet className="h-4 w-4 text-primary" />}
                  />
                  <MetricCard
                    label="有效行数"
                    value={preprocessResult.valid_rows.toLocaleString()}
                    icon={<Rows3 className="h-4 w-4 text-[hsl(var(--success))]" />}
                  />
                  <MetricCard
                    label="过滤行数"
                    value={preprocessResult.removed_rows.toLocaleString()}
                    icon={<Trash2 className="h-4 w-4 text-destructive" />}
                  />
                </div>
                <Button variant="outline" size="sm" className="w-fit gap-2" onClick={() => handleDownloadResult(processedFilename)}>
                  <Download className="h-3.5 w-3.5" /> 下载清洗结果
                </Button>
              </div>
            ) : (
              <EmptyTabState message="第一步完成后在此显示清洗数据。" />
            )}
          </TabsContent>

          {/* Tab 2: Cluster Results */}
          <TabsContent value="tab-cluster" className="mt-4">
            {clusterResult ? (
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <MetricCard
                    label="总文本数"
                    value={clusterResult.total_texts.toLocaleString()}
                    icon={<Layers className="h-4 w-4 text-primary" />}
                  />
                  <MetricCard
                    label="识别类别数"
                    value={clusterResult.n_clusters}
                    icon={<Hash className="h-4 w-4 text-[hsl(var(--success))]" />}
                  />
                  <MetricCard
                    label="噪音数据"
                    value={clusterResult.n_noise.toLocaleString()}
                    icon={<Volume2 className="h-4 w-4 text-destructive" />}
                  />
                </div>
                <Button variant="outline" size="sm" className="w-fit gap-2" onClick={() => handleDownloadResult(realClusteredFilename || fallbackClusteredFilename)}>
                  <Download className="h-3.5 w-3.5" /> 下载聚类结果
                </Button>
              </div>
            ) : (
              <EmptyTabState message="第二步完成后在此显示聚类分布。" />
            )}
          </TabsContent>

          {/* Tab 3: Final Report (核心复刻部分) */}
          <TabsContent value="tab-report" className="mt-4">
            {reportData ? (
              <div className="flex flex-col gap-6">
                {/* 1. 核心指标概览 (Metrics) */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <MetricCard 
                    label="总工单数量" 
                    value={reportData.totalRecords.toLocaleString()} 
                    icon={<Layers className="h-4 w-4 text-muted-foreground" />}
                  />
                  <MetricCard 
                    label="识别议题数" 
                    value={reportData.topicCount} 
                    icon={<Tags className="h-4 w-4 text-[hsl(var(--success))]" />}
                  />
                  <MetricCard 
                    label="议题覆盖率" 
                    value={`${(reportData.coverage * 100).toFixed(1)}%`} 
                    icon={<Target className="h-4 w-4 text-primary" />}
                  />
                  <MetricCard 
                    label="无法归类(噪音)" 
                    value={reportData.noiseCount.toLocaleString()} 
                    icon={<Volume2 className="h-4 w-4 text-destructive" />}
                  />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* 2. Top 10 热点图表 (Charts) */}
                    <Card className="lg:col-span-2 shadow-sm">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-base flex items-center gap-2">
                                <BarChart3 className="h-4 w-4 text-primary" />
                                Top 10 热点诉求排行
                            </CardTitle>
                            <CardDescription>工单数量最多的前 10 个议题</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[300px] w-full mt-2">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={reportData.top10} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e5e7eb" />
                                        <XAxis type="number" hide />
                                        <YAxis 
                                          dataKey="name" 
                                          type="category" 
                                          width={100} 
                                          tick={{fontSize: 12, fill: '#64748b'}} 
                                          axisLine={false}
                                          tickLine={false}
                                        />
                                        <Tooltip 
                                            cursor={{fill: 'transparent'}}
                                            content={({ active, payload }) => {
                                                if (active && payload && payload.length) {
                                                    const data = payload[0].payload;
                                                    return (
                                                        <div className="bg-popover border border-border p-2 rounded-lg shadow-lg text-xs">
                                                            <p className="font-semibold">{data.fullName}</p>
                                                            <p className="text-muted-foreground">数量: {data.count}</p>
                                                        </div>
                                                    );
                                                }
                                                return null;
                                            }}
                                        />
                                        <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={20}>
                                            {reportData.top10.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.fill} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </CardContent>
                    </Card>

                    {/* 3. 快捷下载区 */}
                    <Card className="shadow-sm flex flex-col">
                         <CardHeader className="pb-2">
                            <CardTitle className="text-base">报告导出</CardTitle>
                            <CardDescription>下载分析结果</CardDescription>
                        </CardHeader>
                        <CardContent className="flex flex-col gap-3 flex-1 justify-center">
                            <Button className="w-full gap-2" variant="outline" onClick={() => handleDownloadResult(realKeywordsFilename || fallbackKeywordsFilename)}>
                                <FileSpreadsheet className="h-4 w-4 text-green-600" />
                                下载统计报表 (CSV)
                            </Button>
                            <Button className="w-full gap-2" onClick={() => handleDownloadResult(realKeywordsFilename || fallbackKeywordsFilename)}>
                                <Download className="h-4 w-4" />
                                下载完整明细数据
                            </Button>
                        </CardContent>
                    </Card>
                </div>

                {/* 4. 详细分类统计表 (Table with Progress) */}
                <Card className="shadow-sm">
                  <CardHeader className="py-3 bg-muted/30">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm">详细分类统计表</CardTitle>
                      <Badge variant="secondary" className="text-xs">
                        {reportData.fullDisplayList.length} 类
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-0">
                    <div className="rounded-b-lg border-t overflow-auto max-h-[400px]">
                      <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-background/95 backdrop-blur z-10">
                          <tr className="border-b shadow-sm">
                            <th className="text-left p-3 font-medium text-muted-foreground w-16">ID</th>
                            <th className="text-left p-3 font-medium text-muted-foreground">核心议题 (AI摘要)</th>
                            <th className="text-left p-3 font-medium text-muted-foreground w-24">数量</th>
                            <th className="text-left p-3 font-medium text-muted-foreground w-32">占比</th>
                          </tr>
                        </thead>
                        <tbody>
                          {reportData.fullDisplayList.map((item) => (
                            <tr key={item.clusterId} className="border-b last:border-0 hover:bg-muted/30 transition-colors">
                              <td className="p-3">
                                {item.clusterId === -1 ? (
                                    <Badge variant="destructive" className="text-[10px] px-1 h-5">Noise</Badge>
                                ) : (
                                    <Badge variant="secondary" className="font-mono text-xs">#{item.clusterId}</Badge>
                                )}
                              </td>
                              <td className="p-3 text-foreground font-medium text-xs sm:text-sm">
                                {item.keyword}
                              </td>
                              <td className="p-3 text-muted-foreground tabular-nums">
                                {item.count}
                              </td>
                              <td className="p-3 align-middle">
                                <div className="flex items-center gap-2">
                                    <div className="h-1.5 flex-1 bg-muted rounded-full overflow-hidden min-w-[60px]">
                                        <div 
                                            className={`h-full rounded-full ${item.clusterId === -1 ? 'bg-destructive/50' : 'bg-primary'}`} 
                                            style={{ width: `${(item.percentage * 100).toFixed(1)}%` }}
                                        />
                                    </div>
                                    <span className="text-xs text-muted-foreground w-9 text-right tabular-nums">
                                        {(item.percentage * 100).toFixed(1)}%
                                    </span>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <EmptyTabState message="请先完成第 3 步：大模型智能摘要，生成报告后在此处查看。" />
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Config Dialogs */}
      <ConfigDialog
        open={cleanDialogOpen}
        onOpenChange={setCleanDialogOpen}
        title="数据清洗配置"
        description="选择提取器与目标列"
        onSave={() => setCleanCfg(tempClean)}
      >
        {/* ... Clean Config Content ... */}
        <div className="flex flex-col gap-4 py-2">
            <div className="space-y-2">
                <Label>提取器类型</Label>
                <Select value={tempClean.extractor} onValueChange={(v) => setTempClean(p => ({...p, extractor: v}))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                        <SelectItem value="12366">12366工单提取器</SelectItem>
                        <SelectItem value="12345">12345工单提取器</SelectItem>
                        <SelectItem value="zn">征纳互动提取器</SelectItem>
                    </SelectContent>
                </Select>
            </div>
            <div className="space-y-2">
                <Label>目标列名</Label>
                <Input value={tempClean.column} onChange={(e) => setTempClean(p => ({...p, column: e.target.value}))} />
            </div>
            <div className="flex items-center justify-between border rounded-lg p-3">
                <div className="space-y-0.5">
                    <Label>启用 NER 脱敏</Label>
                    <p className="text-xs text-muted-foreground">推荐开启</p>
                </div>
                <Switch checked={tempClean.useNer} onCheckedChange={(v) => setTempClean(p => ({...p, useNer: v}))} />
            </div>
        </div>
      </ConfigDialog>

      <ConfigDialog
        open={clusterDialogOpen}
        onOpenChange={setClusterDialogOpen}
        title="聚类算法配置"
        description="HDBSCAN 参数调整"
        onSave={() => setClusterCfg(tempCluster)}
      >
        <div className="flex flex-col gap-6 py-2">
            <div className="space-y-2">
                <div className="flex justify-between">
                    <Label>分类精细度 (n_neighbors)</Label>
                    <span className="text-xs text-muted-foreground">{tempCluster.nNeighbors}</span>
                </div>
                <Slider 
                    value={[tempCluster.nNeighbors]} 
                    min={5} max={50} step={1}
                    onValueChange={([v]) => setTempCluster(p => ({...p, nNeighbors: v}))} 
                />
            </div>
            <div className="space-y-2">
                <div className="flex justify-between">
                    <Label>最小成团数 (min_cluster)</Label>
                    <span className="text-xs text-muted-foreground">{tempCluster.minCluster}</span>
                </div>
                <Slider 
                    value={[tempCluster.minCluster]} 
                    min={3} max={50} step={1}
                    onValueChange={([v]) => setTempCluster(p => ({...p, minCluster: v}))} 
                />
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label>关键词数</Label>
                    <Input type="number" value={tempCluster.topN} onChange={(e) => setTempCluster(p => ({...p, topN: +e.target.value}))} />
                </div>
            </div>
        </div>
      </ConfigDialog>

      <ConfigDialog
        open={llmDialogOpen}
        onOpenChange={setLlmDialogOpen}
        title="模型配置"
        description="LLM API 设置"
        onSave={() => setLlmCfg(tempLlm)}
      >
        <div className="flex flex-col gap-4 py-2">
            <div className="space-y-2">
                <Label>API Key</Label>
                <Input type="password" value={tempLlm.apiKey} onChange={(e) => setTempLlm(p => ({...p, apiKey: e.target.value}))} />
            </div>
            <div className="space-y-2">
                <Label>Base URL</Label>
                <Input value={tempLlm.baseUrl} onChange={(e) => setTempLlm(p => ({...p, baseUrl: e.target.value}))} />
            </div>
        </div>
      </ConfigDialog>

      {/* Quick Access */}
      <div className="flex flex-col gap-2 pb-4">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          高级控制台
        </p>
        <div className="flex flex-wrap gap-2">
          {[{ label: "诉求数据清洗 ", page: "preprocess" as PageKey },
            { label: "诉求数据智能聚类", page: "cluster" as PageKey },
            { label: "聚类效果评估", page: "evaluate" as PageKey },
            { label: "诉求数据智能摘要", page: "keywords" as PageKey },
            { label: "结果查看与导出", page: "results" as PageKey }].map((link) => (
            <button
              key={link.page}
              onClick={() => onNavigate(link.page)}
              className="text-xs font-medium px-3 py-1.5 rounded-lg bg-muted text-muted-foreground hover:bg-primary/10 hover:text-primary transition-colors"
            >
              {link.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function EmptyTabState({ message }: { message: string }) {
  return (
    <Card className="border-dashed shadow-none bg-muted/30">
      <CardContent className="py-10">
        <div className="flex flex-col items-center gap-2 text-center">
          <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center">
            <BarChart3 className="h-5 w-5 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground">{message}</p>
        </div>
      </CardContent>
    </Card>
  );
}