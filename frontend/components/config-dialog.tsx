"use client";

import React from "react"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface ConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: React.ReactNode;
  onSave: () => void;
}

export function ConfigDialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  onSave,
}: ConfigDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        <div className="flex flex-col gap-4 py-2">{children}</div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} className="bg-transparent">
            取消
          </Button>
          <Button
            onClick={() => {
              onSave();
              onOpenChange(false);
            }}
          >
            保存并关闭
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
