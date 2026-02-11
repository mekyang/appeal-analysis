"use client";

import { useState } from "react";
import { FileUpload } from "@/components/file-upload";
import { MetricCard } from "@/components/metric-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  clusterData,
  loadState,
  type ClusterResponse,
  type LoadStateResponse,
} from "@/lib/api";
import {
  Brain,
  FolderOpen,
  FileUp,
  Play,
  CheckCircle2,
  AlertCircle,
  Layers,
  Hash,
  Volume2,
  Download,
  Loader2,
  Upload,
} from "lucide-react";

export function ClusterPage() {
  const [file, setFile] = useState<File | null>(null);
  const [textColumn, setTextColumn] = useState("Sanitized_Content");
  const [originalColumn, setOriginalColumn] = useState("业务编号");
  const [nNeighbors, setNNeighbors] = useState(15);
  const [nComponents, setNComponents] = useState(5);
  const [minClusterSize, setMinClusterSize] = useState(10);
  const [keywordTopN, setKeywordTopN] = useState(5);
  const [autoSave, setAutoSave] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingState, setLoadingState] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [result, setResult] = useState<ClusterResponse | null>(null);
  const [stateResult, setStateResult] = useState<LoadStateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleLoadState = async () => {
    setLoadingState(true);
    setError(null);
    try {
      const res = await loadState();
      setStateResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "加载状态失败");
    } finally {
      setLoadingState(false);
    }
  };

  const handleAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      setProgress(15);
      setProgressText("正在上传文件...");
      await new Promise((r) => setTimeout(r, 200));
      setProgress(35);
      setProgressText("SBERT 编码 + HDBSCAN 聚类中...");

      const res = await clusterData(
        file, textColumn, originalColumn,
        nNeighbors, nComponents, minClusterSize, keywordTopN, autoSave
      );
      setProgress(100);
      setProgressText("聚类完成");
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "聚类分析失败");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.output_file) return;
    const filename = result.output_file.split("/").pop();
    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    window.open(`${API_BASE}/api/download/${filename}`, "_blank");
  };

  const displayResult = result || stateResult;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">诉求问题智能聚类 (高级控制台)</h1>
        <p className="text-sm text-muted-foreground mt-1">
          支持上传新文件聚类或加载历史状态恢复
        </p>
      </div>

      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="w-full justify-start bg-muted/50 p-1 rounded-lg">
          <TabsTrigger value="upload" className="gap-1.5 text-xs">
            <Upload className="h-3.5 w-3.5" />
            上传新文件
          </TabsTrigger>
          <TabsTrigger value="history" className="gap-1.5 text-xs">
            <FolderOpen className="h-3.5 w-3.5" />
            加载历史状态
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <FileUp className="h-4 w-4 text-primary" />
                  数据文件
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <FileUpload file={file} onFileChange={setFile} label="上传预处理后的 Excel" />
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1.5">
                    <Label className="text-xs">文本列名</Label>
                    <Input value={textColumn} onChange={(e) => setTextColumn(e.target.value)} className="h-9 text-sm" />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <Label className="text-xs">ID 列名</Label>
                    <Input value={originalColumn} onChange={(e) => setOriginalColumn(e.target.value)} className="h-9 text-sm" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Brain className="h-4 w-4 text-primary" />
                  聚类参数
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">分类精细程度 (n_neighbors)</Label>
                    <Badge variant="secondary" className="text-xs">{nNeighbors}</Badge>
                  </div>
                  <Slider value={[nNeighbors]} onValueChange={([v]) => setNNeighbors(v)} min={5} max={50} step={1} />
                  <p className="text-[10px] text-muted-foreground">值越大，类越少越粗</p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">最小类工单数量 (min_cluster_size)</Label>
                    <Badge variant="secondary" className="text-xs">{minClusterSize}</Badge>
                  </div>
                  <Slider value={[minClusterSize]} onValueChange={([v]) => setMinClusterSize(v)} min={3} max={100} step={1} />
                  <p className="text-[10px] text-muted-foreground">小于此数量归为噪音</p>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col gap-1.5">
                    <Label className="text-xs">n_components</Label>
                    <Input type="number" value={nComponents} onChange={(e) => setNComponents(Number(e.target.value))} min={2} max={10} className="h-9 text-sm" />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <Label className="text-xs">关键词数</Label>
                    <Input type="number" value={keywordTopN} onChange={(e) => setKeywordTopN(Number(e.target.value))} min={1} max={10} className="h-9 text-sm" />
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-2">
                    <Label className="text-xs">自动保存</Label>
                    <Switch checked={autoSave} onCheckedChange={setAutoSave} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="flex items-center gap-3 mt-4">
            <Button onClick={handleAnalysis} disabled={!file || loading} className="gap-2">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {loading ? "分析中..." : "开始完整分析"}
            </Button>
            {result && (
              <Button variant="outline" onClick={handleDownload} className="gap-2 bg-transparent">
                <Download className="h-4 w-4" />
                下载结果
              </Button>
            )}
          </div>
        </TabsContent>

        <TabsContent value="history" className="mt-4">
          <Card>
            <CardContent className="py-8">
              <div className="flex flex-col items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center">
                  <FolderOpen className="h-5 w-5 text-primary" />
                </div>
                <div className="text-center">
                  <h3 className="text-sm font-semibold text-foreground">加载历史状态</h3>
                  <p className="text-xs text-muted-foreground mt-1">恢复上次保存的向量数据与聚类结果</p>
                </div>
                <Button onClick={handleLoadState} disabled={loadingState} className="gap-2">
                  {loadingState ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderOpen className="h-4 w-4" />}
                  {loadingState ? "加载中..." : "加载历史状态"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {loading && (
        <Card>
          <CardContent className="pt-5 pb-5">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{progressText}</span>
                <span className="font-medium text-foreground">{progress}%</span>
              </div>
              <Progress value={progress} className="h-1.5" />
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {displayResult && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-[hsl(var(--success))]" />
            <span className="text-sm font-medium text-foreground">{displayResult.message}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <MetricCard
              label="总文本数"
              value={("total_texts" in displayResult ? displayResult.total_texts : displayResult.total_rows).toLocaleString()}
              icon={<Layers className="h-4 w-4 text-primary" />}
            />
            <MetricCard label="聚类数" value={displayResult.n_clusters} icon={<Hash className="h-4 w-4 text-[hsl(var(--success))]" />} />
            <MetricCard label="噪音数据" value={displayResult.n_noise.toLocaleString()} icon={<Volume2 className="h-4 w-4 text-destructive" />} />
          </div>
        </div>
      )}
    </div>
  );
}
