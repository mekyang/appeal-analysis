"use client";

import { useState, useMemo } from "react";
import { FileUpload } from "@/components/file-upload";
import { MetricCard } from "@/components/metric-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  FileDown,
  AlertCircle,
  Rows3,
  Hash,
  Volume2,
  Upload,
  Table,
} from "lucide-react";
import * as XLSX from "xlsx";

interface RowData {
  [key: string]: string | number | boolean | null | undefined;
}

export function ResultsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<RowData[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [filterNoise, setFilterNoise] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (newFile: File | null) => {
    setFile(newFile);
    setData([]);
    setColumns([]);
    setError(null);
    if (!newFile) return;

    try {
      const buffer = await newFile.arrayBuffer();
      const workbook = XLSX.read(buffer, { type: "array" });
      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json<RowData>(sheet);
      if (jsonData.length === 0) { setError("文件为空"); return; }
      setData(jsonData);
      setColumns(Object.keys(jsonData[0]));
    } catch {
      setError("读取文件失败，请确保文件格式正确");
    }
  };

  const filteredData = useMemo(() => {
    if (!filterNoise || !columns.includes("Cluster")) return data;
    return data.filter((row) => row.Cluster !== -1);
  }, [data, filterNoise, columns]);

  const stats = useMemo(() => {
    if (data.length === 0) return null;
    const hasCluster = columns.includes("Cluster");
    const nClusters = hasCluster
      ? new Set(filteredData.filter((r) => r.Cluster !== -1).map((r) => r.Cluster)).size
      : 0;
    const nNoise = hasCluster ? data.filter((r) => r.Cluster === -1).length : 0;
    return { totalRows: filteredData.length, nClusters, nNoise, hasCluster };
  }, [data, filteredData, columns]);

  const handleExportCSV = () => {
    if (filteredData.length === 0) return;
    const ws = XLSX.utils.json_to_sheet(filteredData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Results");
    XLSX.writeFile(wb, "result_export.xlsx");
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">结果查看与导出</h1>
        <p className="text-sm text-muted-foreground mt-1">
          查看聚类结果数据、过滤噪音、导出文件
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Upload className="h-4 w-4 text-primary" />
              加载文件
            </CardTitle>
          </CardHeader>
          <CardContent>
            <FileUpload file={file} onFileChange={handleFileChange} label="选择结果文件" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <FileDown className="h-4 w-4 text-primary" />
              选项
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex items-center justify-between rounded-lg border p-3">
              <Label className="text-xs">隐藏噪音数据</Label>
              <Switch checked={filterNoise} onCheckedChange={setFilterNoise} />
            </div>
            <Button
              variant="outline"
              onClick={handleExportCSV}
              disabled={filteredData.length === 0}
              className="gap-2 w-full bg-transparent"
            >
              <FileDown className="h-4 w-4" />
              导出 Excel
            </Button>
          </CardContent>
        </Card>
      </div>

      {error && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4">
          <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <MetricCard label="总行数" value={stats.totalRows.toLocaleString()} icon={<Rows3 className="h-4 w-4 text-primary" />} />
          {stats.hasCluster && (
            <>
              <MetricCard label="聚类数" value={stats.nClusters} icon={<Hash className="h-4 w-4 text-[hsl(var(--success))]" />} />
              <MetricCard label="噪音数据" value={stats.nNoise.toLocaleString()} icon={<Volume2 className="h-4 w-4 text-destructive" />} />
            </>
          )}
        </div>
      )}

      {filteredData.length > 0 && (
        <Card>
          <CardHeader className="py-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Table className="h-4 w-4 text-primary" />
                数据预览
              </CardTitle>
              <Badge variant="secondary" className="text-xs">
                显示 {Math.min(filteredData.length, 100)} / {filteredData.length} 行
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="rounded-b-lg border-t overflow-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="sticky top-0">
                  <tr className="bg-muted">
                    {columns.map((col) => (
                      <th key={col} className="text-left p-3 font-medium text-muted-foreground whitespace-nowrap text-xs">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredData.slice(0, 100).map((row, idx) => (
                    <tr key={idx} className="border-t hover:bg-muted/30 transition-colors">
                      {columns.map((col) => (
                        <td key={col} className="p-3 text-foreground max-w-xs truncate text-sm" title={String(row[col] ?? "")}>
                          {col === "Cluster" && row[col] === -1 ? (
                            <Badge variant="destructive" className="text-xs">噪音</Badge>
                          ) : (
                            String(row[col] ?? "")
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {!file && (
        <Card>
          <CardContent className="py-10">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="w-12 h-12 rounded-2xl bg-muted flex items-center justify-center">
                <FileDown className="h-5 w-5 text-muted-foreground" />
              </div>
              <h3 className="text-sm font-medium text-foreground">暂无数据</h3>
              <p className="text-xs text-muted-foreground max-w-sm">
                请上传聚类结果文件 (.xlsx / .xls) 以查看数据
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
