"use client";

import { useState } from "react";
import { FileUpload } from "@/components/file-upload";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { extractKeywords, type KeywordsResponse } from "@/lib/api";
import {
  Tags,
  Key,
  Globe,
  Play,
  CheckCircle2,
  AlertCircle,
  Download,
  Loader2,
  Info,
} from "lucide-react";

export function KeywordsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [apiKey, setApiKey] = useState("sk-f1fec6b90628475ba7ce12b2c389c85a");
  const [baseUrl, setBaseUrl] = useState("https://api.deepseek.com");
  const [textCol, setTextCol] = useState("Text");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<KeywordsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExtract = async () => {
    if (!file || !apiKey) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await extractKeywords(file, apiKey, baseUrl, textCol);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "提取失败");
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

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">大模型智能摘要</h1>
        <p className="text-sm text-muted-foreground mt-1">
          配置 LLM API 进行智能摘要，为每个聚类生成关键词标签
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* API Config */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Key className="h-4 w-4 text-primary" />
              API 配置
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">API Key</Label>
              <Input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className="h-9 text-sm" />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs flex items-center gap-1">
                <Globe className="h-3 w-3" />
                API 地址
              </Label>
              <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} className="h-9 text-sm" />
            </div>
            <div className="flex items-start gap-2 rounded-lg bg-primary/5 border border-primary/10 p-3">
              <Info className="h-3.5 w-3.5 text-primary shrink-0 mt-0.5" />
              <p className="text-xs text-muted-foreground">默认使用内网部署 Qwen 模型</p>
            </div>
          </CardContent>
        </Card>

        {/* Data Source */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Tags className="h-4 w-4 text-primary" />
              数据源
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <FileUpload file={file} onFileChange={setFile} label="上传聚类结果" />
            <div className="flex flex-col gap-1.5 max-w-xs">
              <Label className="text-xs">文本列名</Label>
              <Input value={textCol} onChange={(e) => setTextCol(e.target.value)} className="h-9 text-sm" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action */}
      <div className="flex items-center gap-3">
        <Button onClick={handleExtract} disabled={!file || !apiKey || loading} className="gap-2">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {loading ? "提取中 (请耐心等待)..." : "开始提取关键词"}
        </Button>
        {result && (
          <Button variant="outline" onClick={handleDownload} className="gap-2 bg-transparent">
            <Download className="h-4 w-4" />
            下载结果
          </Button>
        )}
      </div>

      {(!apiKey || !file) && !error && !result && (
        <div className="flex items-start gap-2.5 rounded-xl border border-[hsl(var(--warning))]/30 bg-[hsl(var(--warning))]/5 p-4">
          <AlertCircle className="h-4 w-4 text-[hsl(var(--warning))] shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground">请填写 API 配置并上传聚类结果文件</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {result && result.result.length > 0 && (
        <Card>
          <CardHeader className="py-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-[hsl(var(--success))]" />
              <CardTitle className="text-sm">提取结果</CardTitle>
              <Badge className="bg-[hsl(var(--success))] text-[hsl(var(--success-foreground))] text-xs">
                {result.result.length} 个聚类
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="rounded-b-lg border-t overflow-auto max-h-80">
              <table className="w-full text-sm">
                <thead className="sticky top-0">
                  <tr className="bg-muted/70">
                    <th className="text-left p-3 font-medium text-muted-foreground text-xs w-20">聚类 ID</th>
                    <th className="text-left p-3 font-medium text-muted-foreground text-xs">LLM 关键词</th>
                  </tr>
                </thead>
                <tbody>
                  {result.result.map((item) => (
                    <tr key={item.Cluster} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="p-3">
                        <Badge variant="secondary" className="font-mono text-xs">#{item.Cluster}</Badge>
                      </td>
                      <td className="p-3 text-foreground">{item.LLM_Keywords}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
