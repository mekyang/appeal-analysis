"use client";

import React from "react"

import { cn } from "@/lib/utils";
import { Settings, Play, CheckCircle2, Clock, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

export type StepStatus = "waiting" | "ready" | "running" | "done";

interface PipelineStepCardProps {
  stepNumber: number;
  title: string;
  status: StepStatus;
  statusText?: string;
  onConfigure?: () => void;
  onRun?: () => void;
  runLabel?: string;
  runDisabled?: boolean;
  progress?: number;
  progressText?: string;
  children?: React.ReactNode;
}

const statusConfig: Record<StepStatus, { label: string; className: string; icon: React.ElementType }> = {
  waiting: {
    label: "等待上游数据",
    className: "bg-muted text-muted-foreground border border-border",
    icon: Clock,
  },
  ready: {
    label: "准备就绪",
    className: "bg-primary/10 text-primary border border-primary/20",
    icon: Play,
  },
  running: {
    label: "处理中...",
    className: "bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border border-[hsl(var(--warning))]/20",
    icon: Loader2,
  },
  done: {
    label: "已完成",
    className: "bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border border-[hsl(var(--success))]/20",
    icon: CheckCircle2,
  },
};

export function PipelineStepCard({
  stepNumber,
  title,
  status,
  statusText,
  onConfigure,
  onRun,
  runLabel = "开始处理",
  runDisabled = false,
  progress,
  progressText,
  children,
}: PipelineStepCardProps) {
  const cfg = statusConfig[status];
  const StatusIcon = cfg.icon;

  return (
    <div
      className={cn(
        "flex flex-col rounded-xl border bg-card p-5 transition-all",
        status === "done" && "border-[hsl(var(--success))]/30",
        status === "ready" && "border-primary/30",
        status === "running" && "border-[hsl(var(--warning))]/30"
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-3 pb-3 border-b mb-4">
        <div className="flex items-center justify-center w-7 h-7 rounded-full bg-primary text-primary-foreground text-xs font-bold shrink-0">
          {stepNumber}
        </div>
        <h3 className="text-sm font-semibold text-card-foreground flex-1">{title}</h3>
      </div>

      {/* Status Pill */}
      <div className={cn("flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium w-fit mb-4", cfg.className)}>
        <StatusIcon className={cn("h-3 w-3", status === "running" && "animate-spin")} />
        <span>{statusText || cfg.label}</span>
      </div>

      {/* Content (file upload, etc) */}
      {children && <div className="mb-4">{children}</div>}

      {/* Progress */}
      {status === "running" && progress !== undefined && (
        <div className="flex flex-col gap-1.5 mb-4">
          <Progress value={progress} className="h-1.5" />
          {progressText && (
            <p className="text-xs text-muted-foreground">{progressText}</p>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 mt-auto">
        {onConfigure && (
          <Button
            variant="outline"
            size="sm"
            onClick={onConfigure}
            className="h-8 px-3 text-xs gap-1.5 bg-transparent"
          >
            <Settings className="h-3 w-3" />
            配置
          </Button>
        )}
        {onRun && (
          <Button
            size="sm"
            onClick={onRun}
            disabled={runDisabled || status === "running" || status === "waiting"}
            className="h-8 px-3 text-xs gap-1.5 flex-1"
          >
            {status === "running" ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Play className="h-3 w-3" />
            )}
            {status === "running" ? "处理中..." : runLabel}
          </Button>
        )}
      </div>
    </div>
  );
}
