"use client";

import React from "react"

import { cn } from "@/lib/utils";
import {
  Home,
  FileUp,
  Brain,
  BarChart3,
  Tags,
  FileDown,
  ChevronLeft,
  ChevronRight,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";

export type PageKey =
  | "home"
  | "preprocess"
  | "cluster"
  | "evaluate"
  | "keywords"
  | "results";

interface NavItem {
  key: PageKey;
  label: string;
  shortLabel: string;
  icon: React.ElementType;
  group: string;
}

const navItems: NavItem[] = [
  { key: "home", label: "全流程首页", shortLabel: "首页", icon: Home, group: "概览" },
  { key: "preprocess", label: "诉求问题数据清洗", shortLabel: "清洗", icon: FileUp, group: "流程" },
  { key: "cluster", label: "诉求问题自动聚类", shortLabel: "聚类", icon: Brain, group: "流程" },
  { key: "evaluate", label: "聚类效果评估", shortLabel: "评估", icon: BarChart3, group: "流程" },
  { key: "keywords", label: "诉求问题智能摘要", shortLabel: "摘要", icon: Tags, group: "流程" },
  { key: "results", label: "结果查看与导出", shortLabel: "结果", icon: FileDown, group: "输出" },
];

interface AppSidebarProps {
  currentPage: PageKey;
  onPageChange: (page: PageKey) => void;
  collapsed: boolean;
  onCollapsedChange: (collapsed: boolean) => void;
}

export function AppSidebar({
  currentPage,
  onPageChange,
  collapsed,
  onCollapsedChange,
}: AppSidebarProps) {
  const groups = Array.from(new Set(navItems.map((i) => i.group)));

  return (
    <aside
      className={cn(
        "flex flex-col h-screen bg-[hsl(var(--sidebar-background))] text-[hsl(var(--sidebar-foreground))] border-r border-[hsl(var(--sidebar-border))] transition-all duration-300 shrink-0",
        collapsed ? "w-[60px]" : "w-[240px]"
      )}
    >
      {/* Brand */}
      <div className="flex items-center h-14 px-3 border-b border-[hsl(var(--sidebar-border))]">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-[hsl(var(--sidebar-primary))] text-[hsl(var(--sidebar-primary-foreground))] shrink-0">
            <Zap className="h-4 w-4" />
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <p className="text-sm font-semibold text-[hsl(var(--sidebar-accent-foreground))] truncate leading-tight">
                税费诉求分析
              </p>
              <p className="text-[10px] text-[hsl(var(--sidebar-foreground))]/50 leading-tight">
                v3.0
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 px-2 overflow-y-auto">
        {groups.map((group) => (
          <div key={group} className="mb-2">
            {!collapsed && (
              <p className="text-[10px] font-semibold uppercase tracking-widest text-[hsl(var(--sidebar-foreground))]/40 px-2 py-1.5">
                {group}
              </p>
            )}
            {navItems
              .filter((i) => i.group === group)
              .map((item) => {
                const Icon = item.icon;
                const active = currentPage === item.key;
                return (
                  <button
                    key={item.key}
                    onClick={() => onPageChange(item.key)}
                    title={collapsed ? item.label : undefined}
                    className={cn(
                      "flex items-center gap-2.5 w-full rounded-lg px-2.5 py-2 text-left text-sm transition-all",
                      active
                        ? "bg-[hsl(var(--sidebar-accent))] text-[hsl(var(--sidebar-primary))] font-medium"
                        : "text-[hsl(var(--sidebar-foreground))] hover:bg-[hsl(var(--sidebar-accent))]/60 hover:text-[hsl(var(--sidebar-accent-foreground))]",
                      collapsed && "justify-center px-0"
                    )}
                  >
                    <Icon
                      className={cn(
                        "h-4 w-4 shrink-0",
                        active ? "text-[hsl(var(--sidebar-primary))]" : "text-[hsl(var(--sidebar-foreground))]/60"
                      )}
                    />
                    {!collapsed && (
                      <span className="truncate">{item.label}</span>
                    )}
                  </button>
                );
              })}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-2 border-t border-[hsl(var(--sidebar-border))]">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onCollapsedChange(!collapsed)}
          className="w-full justify-center text-[hsl(var(--sidebar-foreground))]/50 hover:text-[hsl(var(--sidebar-accent-foreground))] hover:bg-[hsl(var(--sidebar-accent))]/50"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>
    </aside>
  );
}
