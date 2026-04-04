import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Syne } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/Providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ClinicalRAG — Multimodal Medical Intelligence",
  description:
    "AI-powered clinical decision support. Research tool only — not for clinical use.",
  keywords: ["medical AI", "clinical decision support", "RAG", "research"],
  robots: "noindex, nofollow",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} ${syne.variable}`}>
      <body className="bg-clinical-950 text-white antialiased font-sans">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
