"use client";

import React from "react"

import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { Upload, FileSpreadsheet, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FileUploadProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  accept?: string;
  label?: string;
  compact?: boolean;
}

export function FileUpload({
  file,
  onFileChange,
  accept = ".xlsx,.xls",
  label = "将 Excel 文件拖放到此处",
  compact = false,
}: FileUploadProps) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) onFileChange(droppedFile);
    },
    [onFileChange]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onFileChange(e.target.files?.[0] ?? null);
    },
    [onFileChange]
  );

  if (file) {
    return (
      <div className="flex items-center gap-3 rounded-xl border bg-card p-3">
        <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 shrink-0">
          <FileSpreadsheet className="h-4 w-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-card-foreground truncate">{file.name}</p>
          <p className="text-xs text-muted-foreground">{(file.size / 1024).toFixed(1)} KB</p>
        </div>
        <Button variant="ghost" size="sm" onClick={() => onFileChange(null)} className="shrink-0 h-8 w-8 p-0">
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed cursor-pointer transition-all",
        compact ? "p-4" : "p-6",
        dragOver
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/40 hover:bg-muted/50"
      )}
    >
      <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10">
        <Upload className="h-4 w-4 text-primary" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">{label}</p>
        <p className="text-xs text-muted-foreground mt-0.5">
          支持 .XLSX / .XLS
        </p>
      </div>
      <input type="file" accept={accept} onChange={handleChange} className="sr-only" />
    </label>
  );
}
