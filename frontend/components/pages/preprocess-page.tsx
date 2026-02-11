"use client";

import { useState } from "react";
import { FileUpload } from "@/components/file-upload";
import { MetricCard } from "@/components/metric-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { preprocessFile, type PreprocessResponse } from "@/lib/api";
import {
  FileUp,
  Settings,
  Play,
  CheckCircle2,
  AlertCircle,
  Download,
  FileSpreadsheet,
  Rows3,
  Trash2,
  Loader2,
} from "lucide-react";

export function PreprocessPage() {
  const [file, setFile] = useState<File | null>(null);
  const [extractorType, setExtractorType] = useState("12366");
  const [useNer, setUseNer] = useState(true);
  const [columnName, setColumnName] = useState("业务内容");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [result, setResult] = useState<PreprocessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      setProgress(10);
      setProgressText("正在读取文件...");
      await new Promise((r) => setTimeout(r, 200));

      setProgress(30);
      setProgressText(`正在使用提取器提取内容...`);

      const res = await preprocessFile(file, extractorType, useNer, columnName);

      setProgress(100);
      setProgressText("处理完成");
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "处理失败");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!file) return;
    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    window.open(`${API_BASE}/api/download/processed_${file.name}`, "_blank");
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">数据接入与清洗 (高级模式)</h1>
        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
          选择提取器类型，配置NER脱敏，处理后可直接下载清洗结果
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* File Upload */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <FileUp className="h-4 w-4 text-primary" />
              文件上传
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FileUpload file={file} onFileChange={setFile} />
            <div className="flex flex-col gap-2">
              <Label className="text-xs">提取器类型</Label>
              <Select value={extractorType} onValueChange={setExtractorType}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="12366">12366 工单提取器</SelectItem>
                  <SelectItem value="12345">12345 工单提取器</SelectItem>
                  <SelectItem value="zn">征纳互动提取器</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Config */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Settings className="h-4 w-4 text-primary" />
              处理配置
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div>
                <Label className="text-sm font-medium">启用NER模型脱敏</Label>
                <p className="text-xs text-muted-foreground mt-0.5">使用 BERT 识别公司名 (非必要不关闭)</p>
              </div>
              <Switch checked={useNer} onCheckedChange={setUseNer} />
            </div>
            <div className="flex flex-col gap-2">
              <Label className="text-xs">输入内容列名</Label>
              <Input value={columnName} onChange={(e) => setColumnName(e.target.value)} />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action bar */}
      <div className="flex items-center gap-3">
        <Button onClick={handleProcess} disabled={!file || loading} className="gap-2">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {loading ? "处理中..." : "开始处理"}
        </Button>
        {result && (
          <Button variant="outline" onClick={handleDownload} className="gap-2 bg-transparent">
            <Download className="h-4 w-4" />
            下载结果
          </Button>
        )}
      </div>

      {/* Progress */}
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

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-[hsl(var(--success))]" />
            <span className="text-sm font-medium text-foreground">{result.message}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <MetricCard label="原始行数" value={result.total_rows.toLocaleString()} icon={<FileSpreadsheet className="h-4 w-4 text-primary" />} />
            <MetricCard label="有效行数" value={result.valid_rows.toLocaleString()} icon={<Rows3 className="h-4 w-4 text-[hsl(var(--success))]" />} />
            <MetricCard label="过滤行数" value={result.removed_rows.toLocaleString()} icon={<Trash2 className="h-4 w-4 text-destructive" />} />
          </div>
        </div>
      )}
    </div>
  );
}
