import React from "react"
import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'

import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: '税务诉求分析系统 Pro',
  description: '基于 SBERT + HDBSCAN 的税务诉求智能聚类分析平台',
}

export const viewport: Viewport = {
  themeColor: '#2563eb',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" className={inter.variable}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}
