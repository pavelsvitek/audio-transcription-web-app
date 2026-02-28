import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "In-Browser Voice Transcription",
  description:
    "Real-time private transcription in Next.js using Whisper Tiny and Web Workers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
