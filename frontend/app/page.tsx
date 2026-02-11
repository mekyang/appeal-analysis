"use client";

import { useState } from "react";
import { AppSidebar, type PageKey } from "@/components/app-sidebar";
import { HomePage } from "@/components/pages/home-page";
import { PreprocessPage } from "@/components/pages/preprocess-page";
import { ClusterPage } from "@/components/pages/cluster-page";
import { EvaluatePage } from "@/components/pages/evaluate-page";
import { KeywordsPage } from "@/components/pages/keywords-page";
import { ResultsPage } from "@/components/pages/results-page";

export default function Page() {
  const [currentPage, setCurrentPage] = useState<PageKey>("home");
  const [collapsed, setCollapsed] = useState(false);

  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage onNavigate={setCurrentPage} />;
      case "preprocess":
        return <PreprocessPage />;
      case "cluster":
        return <ClusterPage />;
      case "evaluate":
        return <EvaluatePage />;
      case "keywords":
        return <KeywordsPage />;
      case "results":
        return <ResultsPage />;
      default:
        return <HomePage onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <AppSidebar
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        collapsed={collapsed}
        onCollapsedChange={setCollapsed}
      />
      <main className="flex-1 overflow-auto">
        <div className="p-6 lg:p-8 max-w-7xl mx-auto">{renderPage()}</div>
      </main>
    </div>
  );
}
