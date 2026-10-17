import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SalesSense AI | Sales Forecasting Dashboard",
  description:
    "AI-powered sales forecasting and trend intelligence using machine learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}