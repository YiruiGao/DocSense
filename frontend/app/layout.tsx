import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: "DocSense - RAG Document Q&A",
  description:
    "A full-stack RAG demo for document ingestion, hybrid retrieval, reranking, grounded answers, and source citations.",
  keywords: [
    "DocSense",
    "RAG",
    "Document Q&A",
    "FastAPI",
    "Next.js",
    "ChromaDB",
    "BM25",
    "Reranking",
  ],
  authors: [{ name: "Gaoyirui" }],
  icons: {
    icon: "/logo.svg",
  },
  openGraph: {
    title: "DocSense - RAG Document Q&A",
    description:
      "Document-grounded Q&A with hybrid retrieval, reranking, and source citations.",
    siteName: "DocSense",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "DocSense - RAG Document Q&A",
    description:
      "Document-grounded Q&A with hybrid retrieval, reranking, and source citations.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground">
        {children}
        <Toaster />
      </body>
    </html>
  );
}
