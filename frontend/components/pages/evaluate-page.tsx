"use client";

import { useState } from "react";
import { FileUpload } from "@/components/file-upload";
import { MetricCard } from "@/components/metric-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { evaluateCluster, type EvaluateResponse } from "@/lib/api";
import {
  BarChart3,
  Play,
  CheckCircle2,
  AlertCircle,
  Activity,
  TrendingUp,
  CircleDot,
  Layers,
  Percent,
  Loader2,
} from "lucide-react";

export function EvaluatePage() {
  const [file, setFile] = useState<File | null>(null);
  const [textColumn, setTextColumn] = useState("Text");
  const [clusterColumn, setClusterColumn] = useState("Cluster");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleEvaluate = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await evaluateCluster(file, textColumn, clusterColumn);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "评估失败");
    } finally {
      setLoading(false);
    }
  };

  const getMetricIcon = (key: string) => {
    if (key.includes("Silhouette")) return <Activity className="h-4 w-4 text-primary" />;
    if (key.includes("CH")) return <TrendingUp className="h-4 w-4 text-[hsl(var(--success))]" />;
    if (key.includes("Cluster") || key.includes("Valid")) return <CircleDot className="h-4 w-4 text-primary" />;
    if (key.includes("Sample") || key.includes("Total")) return <Layers className="h-4 w-4 text-primary" />;
    if (key.includes("Noise") || key.includes("Ratio")) return <Percent className="h-4 w-4 text-destructive" />;
    return <BarChart3 className="h-4 w-4 text-primary" />;
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">聚类效果评估</h1>
        <p className="text-sm text-muted-foreground mt-1">
          轮廓系数、CH 指数、噪音比例等多维度评估
        </p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <BarChart3 className="h-4 w-4 text-primary" />
            评估配置
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <FileUpload file={file} onFileChange={setFile} label="上传聚类结果文件 (须包含 Text 和 Cluster 列)" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">文本列名</Label>
              <Input value={textColumn} onChange={(e) => setTextColumn(e.target.value)} className="h-9 text-sm" />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">聚类列名</Label>
              <Input value={clusterColumn} onChange={(e) => setClusterColumn(e.target.value)} className="h-9 text-sm" />
            </div>
          </div>
          <Button onClick={handleEvaluate} disabled={!file || loading} className="gap-2 w-fit">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? "评估中..." : "开始评估"}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-[hsl(var(--success))]" />
            <span className="text-sm font-medium text-foreground">评估完成</span>
            <Badge className="bg-[hsl(var(--success))] text-[hsl(var(--success-foreground))] text-xs">成功</Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(result.metrics).map(([key, value]) => (
              <MetricCard
                key={key}
                label={key}
                value={typeof value === "number" ? Number(value).toFixed(4) : String(value)}
                icon={getMetricIcon(key)}
              />
            ))}
          </div>

          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-sm">详细指标</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="rounded-b-lg border-t overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-muted/50">
                      <th className="text-left p-3 font-medium text-muted-foreground text-xs">指标名称</th>
                      <th className="text-right p-3 font-medium text-muted-foreground text-xs">数值</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.metrics).map(([key, value]) => (
                      <tr key={key} className="border-t hover:bg-muted/30 transition-colors">
                        <td className="p-3 text-foreground text-sm">{key}</td>
                        <td className="p-3 text-right font-mono text-foreground text-sm">
                          {typeof value === "number" ? Number(value).toFixed(4) : String(value)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
