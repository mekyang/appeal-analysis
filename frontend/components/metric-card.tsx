"use client";

import React from "react"

import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  className?: string;
}

export function MetricCard({ label, value, subtitle, icon, className }: MetricCardProps) {
  return (
    <div className={cn("flex items-start gap-3 rounded-xl border bg-card p-4", className)}>
      {icon && (
        <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 shrink-0 mt-0.5">
          {icon}
        </div>
      )}
      <div className="min-w-0">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <p className="text-xl font-bold text-card-foreground mt-0.5">{value}</p>
        {subtitle && <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>}
      </div>
    </div>
  );
}
